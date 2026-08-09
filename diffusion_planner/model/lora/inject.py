"""InfLoRA injection for the Diffusion Planner DiT cross-attention.

Ports the PLUTO InfLoRA fine-tune mechanism (``src/models/pluto/...``) to
``Diffusion_Planner``. Only the K/V projections of every ``DiTBlock`` cross
attention are adapted with LoRA rank r; the InfLoRA B matrix (dimension
reducer, frozen) is loaded from a precomputed dict and only the expansion A
is trained.

Naming (same twist as PLUTO): the InfLoRA-paper **B** (frozen reducer) lives in
loratorch's ``lora_A`` param; the paper **A** (trained expander) lives in
``lora_B``.  ``mark_only_lora_A_as_trainable`` trains only loratorch ``lora_B``.

Call order contract:  build model -> load base weights -> apply_inflora ->
(DDP wrap) -> build optimizer over ``requires_grad`` params.  Injection must
run AFTER ``Diffusion_Planner`` construction (its ``initialize_weights``
re-inits the DiT) and AFTER base-weight load, but BEFORE the optimizer is
created.
"""
import logging

import torch

from src.models.pluto.layers.utils.util_lora import (
    load_inflora_B_dict,
    mark_only_lora_A_as_trainable,
    replace_mha_modules_with_inflora,
)

logger = logging.getLogger(__name__)

# DiT cross-attn modules, relative to a Diffusion_Planner root:
#   Diffusion_Planner.decoder (Diffusion_Planner_Decoder)
#     .decoder (Decoder) .dit (DiT) .blocks[i] (DiTBlock) .cross_attn
DEFAULT_DIFFUSION_TARGETS = ["decoder.decoder.dit.blocks.*.cross_attn"]


def load_base_weights(model, path, prefer_ema=False, device="cpu"):
    """Load base-model weights (weights only) into a raw ``Diffusion_Planner``.

    Handles the checkpoint layout produced by ``save_model`` (a dict with
    ``'model'`` / ``'ema_state_dict'``) as well as a bare ``state_dict``, and
    strips a ``'module.'`` prefix left by DDP.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema_state_dict" in ckpt):
        if prefer_ema and "ema_state_dict" in ckpt:
            sd = ckpt["ema_state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt["ema_state_dict"]
    else:
        sd = ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        "load_base_weights(%s): loaded %d keys (missing=%d, unexpected=%d)",
        path, len(sd), len(missing), len(unexpected),
    )
    return model


def _freeze_query_rows_of_kv_lora(model, replaced_names):
    """Zero + gradient-freeze the query (Q) output rows of every trainable
    ``kv_lora_B`` so InfLoRA adapts K/V projections only.

    in_proj_weight out-dim = 3*embed  (Q=[0:embed], K=[embed:2*embed], V=[2*embed:]).
    kv_lora_B has shape (3*embed, r); we zero rows [0:embed] and register a backward
    hook returning a gradient with those rows zeroed → Q rows never update (stay 0),
    equivalent to freezing them (device-safe: the hook zeros on the grad's own device).
    """
    name2mod = dict(model.named_modules())
    n = 0
    for name in replaced_names:
        module = name2mod.get(name)
        lora_B = getattr(module, "kv_lora_B", None) if module is not None else None
        if lora_B is None:
            continue
        out = lora_B.shape[0]
        if out % 3 != 0:
            logger.warning("kv_lora_B out=%d not divisible by 3 in %s; skip Q mask", out, name)
            continue
        embed = out // 3
        with torch.no_grad():
            lora_B[:embed].zero_()                     # Q rows → 0 (ΔW[Q]=0 at start)
        lora_B.register_hook(lambda grad, e=embed: _zero_query_grad(grad, e))
        n += 1
    return n


def _zero_query_grad(grad, embed):
    g = grad.clone()
    g[:embed] = 0                                       # no update to Q rows
    return g


def apply_inflora_to_diffusion(model, args):
    """Replace DiT cross-attn MHA with InfLoRA and freeze all but the A matrices.

    Reads from ``args``: ``lora_r``, ``lora_alpha`` (None -> r), ``inflora_B_path``,
    ``lora_target_modules`` (None -> DEFAULT_DIFFUSION_TARGETS), ``pure_lora``.

    Mirrors the PlanningModel InfLoRA build (``pluto_model.py`` lines 121-162).
    Returns the list of replaced module names.
    """
    lora_r = int(args.lora_r)
    lora_alpha = getattr(args, "lora_alpha", None)
    lora_alpha = int(lora_alpha) if lora_alpha is not None else lora_r
    targets = getattr(args, "lora_target_modules", None) or DEFAULT_DIFFUSION_TARGETS
    pure_lora = bool(getattr(args, "pure_lora", False))

    if pure_lora:
        # Plain LoRA baseline: NO frozen SVD B; lora_A random, lora_B zero-init
        # (ΔW=0 at start); BOTH matrices trained.
        from loratorch.utils import mark_only_lora_as_trainable

        logger.info("Pure LoRA (no InfLoRA B): r=%d alpha=%d scaling=%.3f",
                    lora_r, lora_alpha, lora_alpha / lora_r)
        replaced = replace_mha_modules_with_inflora(
            model, target_modules=targets, lora_r=lora_r,
            lora_alpha=lora_alpha, inflora_B_dict=None,
        )
        mark_only_lora_as_trainable(model)
    else:
        B_path = getattr(args, "inflora_B_path", None)
        if B_path is None:
            raise ValueError(
                "inflora_B_path must be set when use_lora_fine_tune=True and "
                "pure_lora=False. Run tool/inflora/prepare_inflora_diffusion.py "
                "to compute B matrices, or pass --pure_lora for a plain-LoRA baseline."
            )
        logger.info("InfLoRA B from %s: r=%d alpha=%d scaling=%.3f",
                    B_path, lora_r, lora_alpha, lora_alpha / lora_r)
        B_dict = load_inflora_B_dict(B_path)
        replaced = replace_mha_modules_with_inflora(
            model, target_modules=targets, lora_r=lora_r,
            lora_alpha=lora_alpha, inflora_B_dict=B_dict,
        )
        mark_only_lora_A_as_trainable(model)

    if len(replaced) == 0:
        raise RuntimeError(
            f"apply_inflora_to_diffusion replaced 0 modules. Check that target "
            f"patterns {targets} match names in model.named_modules() "
            f"(expected e.g. 'decoder.decoder.dit.blocks.0.cross_attn')."
        )

    # --- true K/V-only masking (query gradient freeze) ---
    # loratorch registers the "kv" LoRA on the FULL in_proj_weight (out = 3*embed),
    # so the trainable kv_lora_B spans Q|K|V output rows. Without this, gradient
    # leaks into the Q rows and Q gets adapted with the cross_c-derived B subspace
    # (which is only valid for K/V, whose input IS cross_c; Q's input is x).
    # We zero the Q rows now and register a backward hook that zeros their gradient
    # every step, so the Q rows stay exactly 0 through training AND merge → K/V only.
    n_masked = _freeze_query_rows_of_kv_lora(model, replaced)
    if n_masked:
        logger.info("InfLoRA K/V-only: froze query rows of kv_lora_B in %d modules", n_masked)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("InfLoRA applied to %d modules; trainable=%d / total=%d params",
                len(replaced), n_train, n_total)
    return replaced
