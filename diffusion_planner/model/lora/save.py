"""LoRA-aware checkpoint saving for the Diffusion Planner.

Two artifacts, mirroring the PLUTO ``LoRABaseFormatCheckpointCallback``:

  ``save_lora_ab``       -> ``{module: {"B__<pn>": frozen reducer,
                                        "A__<pn>": trained expander}}``
                            (loratorch lora_A = paper B, lora_B = paper A).
  ``export_merged_base`` -> an inference-ready checkpoint (same layout as
                            ``save_model``) with ΔW merged into ``in_proj_weight``
                            and all ``lora_*`` keys removed.  Drop-in for
                            ``planner.py`` / ``diffusion_adapter.py`` loading.

Handles the loratorch ``in_proj_weight`` demotion bug: ``loratorch.forward``
replaces ``in_proj_weight`` with a plain tensor, so after a no-backward pass it
is missing from ``state_dict()``.  We call ``register_weight_after_backward``
before reading the state dict.
"""
import logging

import torch

from src.models.pluto.layers.transformer_lora import InfLoRAMultiheadAttention
from src.models.pluto.layers.utils.util_lora import merge_lora_into_base

logger = logging.getLogger(__name__)


def _lora_modules(model):
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, InfLoRAMultiheadAttention)
    }


def register_lora_weights(model):
    """Re-register any ``in_proj_weight`` that ``loratorch.forward`` demoted to a
    plain tensor, so it reappears in ``state_dict()``.

    Must be called after ``loss.backward()`` and BEFORE anything that reads
    ``model.state_dict()`` — notably ``timm.ModelEma.update`` (which otherwise
    raises ``KeyError: ...in_proj_weight``) and any checkpoint save.  No-op when
    the model has no InfLoRA modules.
    """
    mods = _lora_modules(model)
    if not mods:
        return
    with torch.no_grad():
        for m in mods.values():
            m.register_weight_after_backward()


def save_lora_ab(model, path):
    """Dump per-module LoRA A/B matrices.

    loratorch ``lora_A`` = InfLoRA-paper B (frozen dim-reducer) -> stored as ``B__*``;
    loratorch ``lora_B`` = InfLoRA-paper A (trained expander)   -> stored as ``A__*``.
    """
    mods = _lora_modules(model)
    if not mods:
        logger.warning("save_lora_ab: no InfLoRA modules found; nothing saved")
        return
    ab = {}
    for name, mod in mods.items():
        entry = {}
        for pn, p in mod.named_parameters():
            if "lora_A" in pn:
                entry[f"B__{pn}"] = p.data.detach().cpu().clone()
            elif "lora_B" in pn:
                entry[f"A__{pn}"] = p.data.detach().cpu().clone()
        if entry:
            ab[name] = entry
    torch.save(ab, path)
    logger.info("save_lora_ab: %d modules -> %s", len(ab), path)


def _merged_state_dict(root):
    """``root.state_dict()`` with ΔW manually merged into ``in_proj_weight`` and
    all ``lora_*`` keys removed.  Non-destructive (the model is not modified), so
    safe to call mid-training.  Key prefixes (e.g. ``module.``) are preserved.

    Manual merge (base + scaling · lora_B @ lora_A) instead of loratorch's
    ``merge_lora_into_base``.  Root cause of the old EMA bug: ``timm.ModelEma``
    puts its shadow in eval mode at init (``self.ema.eval()``), which sets each
    InfLoRA module's ``merged=True``.  ``merge_lora_into_base`` triggers the merge
    via ``model.eval() -> add_lora_data``, gated by ``if not merged`` — so for the
    EMA shadow (always ``merged=True``) the add is SKIPPED and the tracked
    ``lora_B`` is never folded (``ema_state_dict`` ends up ≈ base).  loratorch's
    ``forward`` also demotes ``in_proj_weight`` to a plain tensor, making the
    stateful merge unreliable after several forward passes.  The manual path is
    deterministic and independent of the ``merged`` flag and demotion state:
    ``in_proj_weight`` here is always the (frozen) base — the live model keeps it
    unmerged in train mode, and the EMA shadow tracks that same base — so adding
    ``merge_BA * scaling`` once yields exactly base + ΔW.
    """
    mods = _lora_modules(root)
    if not mods:
        return {k: v.detach().cpu() for k, v in root.state_dict().items()}

    # Re-register in_proj_weight demoted by loratorch.forward (else it is missing
    # from state_dict / getattr after a no-backward pass).
    with torch.no_grad():
        for m in mods.values():
            m.register_weight_after_backward()

    # Manually compute base + ΔW per module (merge_BA gives the correctly shaped
    # /transposed lora_B @ lora_A; * scaling matches loratorch's merge_lora_param).
    merged = {}
    with torch.no_grad():
        for name, m in mods.items():
            for pn in m.params_with_lora:  # e.g. 'in_proj_weight'
                base = getattr(m, pn).detach()
                delta = m.merge_BA(pn) * m.scaling
                merged[f"{name}.{pn}"] = (base + delta).detach().cpu()

    out = {}
    for k, v in root.state_dict().items():
        if "lora_A" in k or "lora_B" in k:
            continue
        mk = next((nm for nm in merged if k.endswith(nm)), None)
        out[k] = merged[mk] if mk is not None else v.detach().cpu()
    return out


def export_merged_base(model, path, ema=None):
    """Save an inference-ready checkpoint with LoRA merged into the base weights.

    ``model`` should be the object passed to ``save_model`` (i.e. the
    DDP-wrapped training model), so state-dict keys keep the ``module.`` prefix
    that ``planner.py`` filters on.  ``ema`` is the timm ``ModelEma`` wrapper
    (its ``.ema`` shadow is merged into ``ema_state_dict``).
    """
    out = {"model": _merged_state_dict(model)}
    if ema is not None:
        out["ema_state_dict"] = _merged_state_dict(ema.ema)
    torch.save(out, path)
    logger.info("export_merged_base -> %s (ema=%s)", path, ema is not None)
