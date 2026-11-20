from __future__ import annotations

from abc import ABC, abstractmethod

import fcl
import numpy as np
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from ompl import base as ob
from ompl import geometric as og
from scipy.spatial.transform import Rotation

from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation_components.constraints import (
    PayloadConstraints,
    PayloadStateValidator,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation_components.models import (
    PlannedPath,
    PlannedRoute,
    Waypoint,
    WaypointRoute,
    WorkspaceBounds,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation_components.optimisation_objective import (
    BlanketTiltObjective,
    MaximizeCenterlineTranslationObjective,
)


class Planner(ABC):
    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def plan(self, *args, **kwargs):
        pass


class SinglePathPlanner(Planner):
    @staticmethod
    def create_space(workspace_bounds: WorkspaceBounds) -> ob.SE3StateSpace:
        space = ob.SE3StateSpace()

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, workspace_bounds.minimum_x)
        bounds.setHigh(0, workspace_bounds.maximum_x)
        bounds.setLow(1, workspace_bounds.minimum_y)
        bounds.setHigh(1, workspace_bounds.maximum_y)
        bounds.setLow(2, workspace_bounds.minimum_z)
        bounds.setHigh(2, workspace_bounds.maximum_z)

        space.setBounds(bounds)
        return space

    @staticmethod
    def pose_matrix(state, payload_pivot_point: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        rotation = Rotation.from_quat([
            state.rotation().x,
            state.rotation().y,
            state.rotation().z,
            state.rotation().w,
        ]).as_matrix()
        position = np.array([state.getX(), state.getY(), state.getZ()])
        matrix[:3, :3] = rotation
        matrix[:3, 3] = position - rotation @ payload_pivot_point
        return matrix

    def __init__(
        self,
        payload: list[fcl.CollisionObject],
        payload_pivot_point: np.ndarray,
        payload_constraints: PayloadConstraints,
        environment: list[tuple[str, np.ndarray, fcl.CollisionObject]],
        workspace_bounds: WorkspaceBounds,
        centerline_direction: np.ndarray,
    ):
        self.payload = payload
        self.environment = environment
        self.payload_pivot_point = payload_pivot_point
        self.payload_constraints = payload_constraints
        self.workspace_bounds = workspace_bounds
        self.centerline_direction = centerline_direction

    def create_validator(self) -> PayloadStateValidator:
        return PayloadStateValidator(
            self.payload,
            self.environment,
            self.payload_constraints,
            self.payload_pivot_point,
        )

    def create_state(self, space: ob.SE3StateSpace, waypoint: Waypoint):
        state = space.allocState()
        state.setX(float(waypoint.position[0]))
        state.setY(float(waypoint.position[1]))
        state.setZ(float(waypoint.position[2]))
        quaternion = waypoint.quaternion
        state.rotation().x = float(quaternion[0])
        state.rotation().y = float(quaternion[1])
        state.rotation().z = float(quaternion[2])
        state.rotation().w = float(quaternion[3])
        return state

    def create_space_information(
        self, validator: PayloadStateValidator | None = None
    ) -> tuple[ob.SE3StateSpace, ob.SpaceInformation]:
        space = self.create_space(self.workspace_bounds)
        space_information = ob.SpaceInformation(space)
        space_information.setStateValidityChecker(
            self.create_validator() if validator is None else validator
        )
        space_information.setStateValidityCheckingResolution(0.002)  # CONFIGURABLE
        space_information.setup()
        return (space, space_information)

    def validate(self, waypoint: Waypoint) -> None:
        validator = self.create_validator()
        validator.diagnostic_waypoint_name = waypoint.name
        space, space_information = self.create_space_information(validator)
        state = self.create_state(space, waypoint)
        if not space_information.isValid(state):
            reason = (
                ""
                if validator.last_invalid_reason is None
                else f" ({validator.last_invalid_reason})"
            )
            raise RuntimeError(f"Invalid waypoint: {waypoint.name}{reason}")

    def plan(
        self, start_waypoint: Waypoint, goal_waypoint: Waypoint, timeout_s: float = 15.0
    ) -> PlannedPath:
        space, space_information = self.create_space_information()
        start = self.create_state(space, start_waypoint)
        goal = self.create_state(space, goal_waypoint)

        if not space_information.isValid(start):
            raise RuntimeError(f"Start waypoint invalid: {start_waypoint.name}")

        if not space_information.isValid(goal):
            raise RuntimeError(f"Goal waypoint invalid: {goal_waypoint.name}")

        problem = ob.ProblemDefinition(space_information)
        problem.setStartAndGoalStates(start, goal)

        multi = ob.MultiOptimizationObjective(space_information)
        tilt_objective = BlanketTiltObjective(space_information)
        centerline_objective = MaximizeCenterlineTranslationObjective(
            space_information, self.centerline_direction
        )
        multi.addObjective(tilt_objective)
        # TODO: The principle of this is that the port is widest towards the back and so we should move there first.
        # In practice this doesn't really work as intended for OBS1. To be reviewed and revised.
        # multi.addObjective(centerline_objective, weight=100000)
        problem.setOptimizationObjective(multi)

        planner = og.RRTConnect(space_information)
        planner.setRange(0.1)  # CONFIGURABLE
        planner.setProblemDefinition(problem)
        planner.setup()

        solved = planner.solve(timeout_s)
        path = problem.getSolutionPath()

        if not solved or path is None or path.getStateCount() == 0:
            raise RuntimeError(
                f"No valid solution found: {start_waypoint.name} -> {goal_waypoint.name}"
            )

        try:
            simplifier = og.PathSimplifier(space_information)
            simplifier.collapseCloseVertices(path)
            simplifier.reduceVertices(path)

        except Exception as error:
            bluemira_warn(f"Path simplification failed: {error}")

        return PlannedPath(path=path, start_state=start, goal_state=goal)


class RoutePlanner(Planner):
    def __init__(self, planner: SinglePathPlanner):
        self.planner = planner

    def validate(self, route: WaypointRoute) -> None:
        for waypoint in route.waypoints:
            self.planner.validate(waypoint)

    def plan(
        self,
        route: WaypointRoute,
        timeout=15,
        initial_world_rotation: np.ndarray | None = None,
    ) -> PlannedRoute:
        self.validate(route)
        segment_paths = []
        for index in range(len(route.waypoints) - 1):
            start_waypoint = route.waypoints[index]
            goal_waypoint = route.waypoints[index + 1]
            bluemira_print(f"Planning {start_waypoint.name} -> {goal_waypoint.name}")
            segment_paths.append(
                self.planner.plan(
                    start_waypoint, goal_waypoint, timeout / len(route.waypoints)
                )
            )

        return PlannedRoute(
            segment_paths=segment_paths,
            initial_world_rotation=initial_world_rotation,
        )
