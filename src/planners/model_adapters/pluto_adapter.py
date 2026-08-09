"""PLUTO adapter: runs `PlanningModel` and returns its single ego trajectory.

Upstream PLUTO also emits top-k candidates, their probabilities, agent
predictions and a reference-free trajectory. `RefinementPlanner` deliberately
uses none of that — see CLAUDE.md "What was deliberately removed" — so only
`output_trajectory` is unpacked here.
"""

from typing import Optional

import torch

from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from ..ml_planner_utils import load_checkpoint
from .base_adapter import AdapterOutput, PlannerModelAdapter


class PlutoModelAdapter(PlannerModelAdapter):
    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: Optional[str] = None,
    ) -> None:
        # `planner_ckpt` is the common checkpoint field across all adapters
        # (following the existing PLUTO baseline naming), so a single
        # `model_adapter.planner_ckpt=...` override works regardless of which
        # adapter (pluto/diffusion) is selected.
        self._planner = planner
        self._ckpt_path = planner_ckpt
        self._feature_builder = planner.get_list_of_required_feature()[0]
        self._device = torch.device("cpu")

    @property
    def feature_builder(self):
        return self._feature_builder

    def initialize(
        self,
        initialization: PlannerInitialization,
        device: torch.device,
    ) -> None:
        self._device = device
        if self._ckpt_path is not None:
            self._planner.load_state_dict(load_checkpoint(self._ckpt_path))
        self._planner.eval()
        self._planner = self._planner.to(device)

    @torch.no_grad()
    def build_and_forward(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> AdapterOutput:
        planner_feature = self._feature_builder.get_features_from_simulation(
            current_input, initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor()]
        ).to_device(self._device)

        out = self._planner.forward(planner_feature_torch.data)

        return AdapterOutput(
            ml_local=out["output_trajectory"][0],
            data=planner_feature_torch.data,
        )
