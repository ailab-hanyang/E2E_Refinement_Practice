"""Diffusion adapter: drives `RefinementPlanner` with the Diffusion planner model.

Hybrid design: the adapter builds **two** things per step.

1. Diffusion features via the diffusion package's own `DataProcessor`, used only
   to run `Diffusion_Planner.forward` and produce the ego ML trajectory.
2. A PLUTO-style geometry `data` dict via the existing `PlutoFeatureBuilder`,
   reused unchanged by the MPC initial state and lane-boundary generation —
   neither of which can consume diffusion's tensor layout.

Both builders anchor on the ego rear-axle frame, so the two views are consistent.
The cost is building features twice per step.
"""

import warnings

import numpy as np
import torch

from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .base_adapter import AdapterOutput, PlannerModelAdapter

FUTURE_STEPS = 80


class DiffusionModelAdapter(PlannerModelAdapter):

    def __init__(
        self,
        planner_ckpt: str,
        args_file: str,
        geometry_feature_builder: PlutoFeatureBuilder,
        guidance_fn=None,
        past_trajectory_sampling: TrajectorySampling = None,
        future_trajectory_sampling: TrajectorySampling = None,
        enable_ema: bool = True,
    ) -> None:
        # `planner_ckpt` is the diffusion .pth (the only artifact you swap between
        # runs; common field name across adapters). `args_file` (invariant
        # inference-time normalization stats, built once at train time) is pinned
        # in the YAML config, not on the command line.
        self._ckpt_path = planner_ckpt
        config = Config(args_file, guidance_fn)

        self._config = config
        self._geometry_feature_builder = geometry_feature_builder
        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling
        self._enable_ema = enable_ema

        self._device = torch.device("cpu")
        self._device_str = "cpu"
        self._map_api = None
        self._route_roadblock_ids = None

        self._planner = Diffusion_Planner(config)
        self._data_processor = DataProcessor(config)
        self._observation_normalizer = config.observation_normalizer

    @property
    def feature_builder(self):
        return self._geometry_feature_builder

    def build_dagger_sample(self, scenario, iteration, ego_state_history, refined_trajectory,
                            ml_trajectory=None):
        """Assemble one diffusion-format `.npz` training sample for a DAgger cache hit.

        Delegates to the diffusion `DataProcessor`. `ego_state_history` is the
        rollout past and `refined_trajectory` is the MPC-refined pseudo-GT future
        (both from the planner). Returns a dict of numpy arrays, or None.
        """
        return self._data_processor.build_dagger_sample(
            scenario, iteration, ego_state_history, refined_trajectory,
            ml_trajectory=ml_trajectory,
        )

    def initialize(
        self,
        initialization: PlannerInitialization,
        device: torch.device,
    ) -> None:
        self._device = device
        self._device_str = "cuda" if device.type == "cuda" else "cpu"
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict = torch.load(self._ckpt_path, map_location=self._device_str)
            if self._enable_ema:
                state_dict = state_dict["ema_state_dict"]
            elif "model" in state_dict.keys():
                state_dict = state_dict["model"]
            # DDP-trained checkpoints carry a "module." prefix; strip it. Fall
            # back to the raw dict if nothing was prefixed (single-GPU ckpt).
            stripped = {
                k[len("module."):]: v
                for k, v in state_dict.items()
                if k.startswith("module.")
            }
            self._planner.load_state_dict(stripped if stripped else state_dict)
        else:
            warnings.warn("DiffusionModelAdapter: no ckpt_path, using random weights")

        self._planner.eval()
        self._planner = self._planner.to(self._device)

    @torch.no_grad()
    def build_and_forward(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> AdapterOutput:
        # (1) PLUTO-style geometry for MPC / boundary / TTC.
        geom_feature = self._geometry_feature_builder.get_features_from_simulation(
            current_input, initialization
        )
        geom_torch = geom_feature.collate(
            [geom_feature.to_feature_tensor()]
        ).to_device(self._device)

        # (2) Diffusion features + forward → ego ML trajectory.
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        inputs = self._data_processor.observation_adapter(
            history,
            traffic_light_data,
            self._map_api,
            self._route_roadblock_ids,
            self._device_str,
        )
        inputs = self._observation_normalizer(inputs)
        _, dec = self._planner(inputs)

        pred = dec["prediction"][0, 0]  # (T, 4) ego-local [x, y, cos, sin]
        heading = torch.atan2(pred[:, 3], pred[:, 2]).unsqueeze(-1)
        ml_local = torch.cat([pred[:, :2], heading], dim=-1)  # (T, 3) [x, y, yaw]
        ml_local = self._ensure_horizon(ml_local)

        return AdapterOutput(
            ml_local=ml_local,
            data=geom_torch.data,
        )

    @staticmethod
    def _ensure_horizon(ml_local: torch.Tensor) -> torch.Tensor:
        """The whole MPC/DAgger pipeline hardcodes an 80-step, 0.1s horizon.
        Resample the diffusion trajectory to 80 steps if it differs."""
        t = ml_local.shape[0]
        if t == FUTURE_STEPS:
            return ml_local
        src = torch.linspace(0.0, 1.0, t, device=ml_local.device)
        dst = torch.linspace(0.0, 1.0, FUTURE_STEPS, device=ml_local.device)
        cols = []
        for c in range(ml_local.shape[1]):
            # np.interp for a robust 1-D linear resample (yaw already unwrapped
            # per-step; small interpolation error is acceptable pre-MPC).
            cols.append(
                torch.from_numpy(
                    np.interp(dst.cpu().numpy(), src.cpu().numpy(), ml_local[:, c].cpu().numpy())
                ).to(ml_local.device, ml_local.dtype)
            )
        return torch.stack(cols, dim=-1)
