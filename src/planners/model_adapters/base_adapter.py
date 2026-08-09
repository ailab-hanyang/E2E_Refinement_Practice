"""Model-agnostic adapter interface for `RefinementPlanner`.

The planner must not know which model produced the trajectory it refines, so it
only ever talks to a `PlannerModelAdapter`. Each concrete adapter is responsible
for:

* building whatever features its model needs and running a forward pass, and
* returning an `AdapterOutput` with the two things the planner consumes: the ego
  ML trajectory, and the geometry dict the MPC preprocessing reads.

Everything downstream of the adapter (MPC refinement, scoring, rendering) is
numpy / global frame and model-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch

from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)


@dataclass
class AdapterOutput:
    """Everything `RefinementPlanner` needs from a model for one planning step.

    Both fields are already batch-stripped (batch index 0 removed).
    """

    # Ego ML trajectory (T, 3) in ego-local frame, ordered [x, y, yaw]. Torch
    # tensor on the planner device so it can feed
    # `TrajectoryVelocityInfoGeneratorTorch` directly.
    ml_local: torch.Tensor

    # Collated, on-device feature dict. Supplies `current_state` for the MPC
    # initial state and `map` for lane-boundary generation.
    data: Dict[str, Any]


class PlannerModelAdapter(ABC):
    """Interface `RefinementPlanner` uses to stay model-agnostic."""

    @abstractmethod
    def initialize(
        self,
        initialization: PlannerInitialization,
        device: torch.device,
    ) -> None:
        """Load weights and move the model to ``device``.

        Called from `RefinementPlanner.initialize`. ``initialization`` carries the
        map API and route roadblock ids some feature builders need.
        """

    @abstractmethod
    def build_and_forward(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> AdapterOutput:
        """Build features, run the model, and return a normalized `AdapterOutput`."""

    @property
    @abstractmethod
    def feature_builder(self):
        """The feature builder whose `scenario_manager` the planner wires up, so
        the model and the planner share one route / drivable-area view."""
