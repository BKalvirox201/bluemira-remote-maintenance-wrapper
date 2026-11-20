from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def _normalise_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    normalised = np.array(rotation_matrix, dtype=float, copy=True)
    normalised[np.abs(normalised) < 1e-12] = 0.0
    return Rotation.from_matrix(normalised).as_matrix()


def euler_degrees_to_rotation_matrix(
    yaw_degrees: float, pitch_degrees: float, roll_degrees: float
) -> np.ndarray:
    return _normalise_rotation_matrix(
        Rotation.from_euler(
            "ZYX",
            [yaw_degrees, pitch_degrees, roll_degrees],
            degrees=True,
        ).as_matrix()
    )


def rotation_matrix_to_euler_degrees(
    rotation_matrix: np.ndarray,
) -> tuple[float, float, float]:
    yaw_degrees, pitch_degrees, roll_degrees = Rotation.from_matrix(
        _normalise_rotation_matrix(rotation_matrix)
    ).as_euler("ZYX", degrees=True)
    return float(yaw_degrees), float(pitch_degrees), float(roll_degrees)


def rotation_matrix_to_world_euler_degrees(
    rotation_matrix: np.ndarray,
) -> tuple[float, float, float]:
    yaw_degrees, pitch_degrees, roll_degrees = Rotation.from_matrix(
        _normalise_rotation_matrix(rotation_matrix)
    ).as_euler("zyx", degrees=True)
    return float(yaw_degrees), float(pitch_degrees), float(roll_degrees)


@dataclass
class Mesh:
    name: str
    vertices: NDArray[np.float64]
    triangles: NDArray[np.int32]
    pivot_point: NDArray[np.float64]
    initial_rotation: NDArray[np.float64] | None = None


@dataclass
class WorkspaceBounds:
    minimum_x: float
    maximum_x: float
    minimum_y: float
    maximum_y: float
    minimum_z: float
    maximum_z: float


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: np.ndarray
    roll_degrees: float = 0.0
    pitch_degrees: float = 0.0
    yaw_degrees: float = 0.0
    rotation_matrix: np.ndarray | None = None
    is_staging: bool = False

    def __post_init__(self) -> None:
        if self.rotation_matrix is None:
            matrix = euler_degrees_to_rotation_matrix(
                self.yaw_degrees,
                self.pitch_degrees,
                self.roll_degrees,
            )
            object.__setattr__(self, "rotation_matrix", matrix)

    @property
    def quaternion(self) -> np.ndarray:
        return Rotation.from_matrix(self.rotation_matrix).as_quat()


@dataclass
class WaypointRoute:
    waypoints: list[Waypoint]

    @property
    def start(self) -> Waypoint:
        return self.waypoints[0]

    @property
    def goal(self) -> Waypoint:
        return self.waypoints[-1]

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("Waypoint route requires at least two waypoints")


@dataclass
class PlannedPath:
    path: object
    start_state: object
    goal_state: object

    def states(self):
        for index in range(self.path.getStateCount()):
            yield self.path.getState(index)


@dataclass
class PlannedRoute:
    segment_paths: list[PlannedPath]
    initial_world_rotation: np.ndarray | None = None

    def states(self):
        for segment_index, segment in enumerate(self.segment_paths):
            for state_index, state in enumerate(segment.states()):
                if segment_index > 0 and state_index == 0:
                    continue
                yield state

    def to_human_readable_string(self) -> str:
        lines = ["Planned route (blanket orientation in world frame):"]
        state_number = 0
        initial_rotation = (
            np.eye(3)
            if self.initial_world_rotation is None
            else self.initial_world_rotation
        )

        for segment_number, segment in enumerate(self.segment_paths, start=1):
            lines.append(f"  Segment {segment_number}:")
            for state_number_in_segment, state in enumerate(segment.states()):
                if segment_number > 1 and state_number_in_segment == 0:
                    continue

                position = np.array([state.getX(), state.getY(), state.getZ()])
                rotation = Rotation.from_quat([
                    state.rotation().x,
                    state.rotation().y,
                    state.rotation().z,
                    state.rotation().w,
                ])
                yaw, pitch, roll = rotation_matrix_to_world_euler_degrees(
                    rotation.as_matrix() @ initial_rotation
                )
                lines.append(
                    f"    {state_number}: position=({position[0]:.3f}, "
                    f"{position[1]:.3f}, {position[2]:.3f}), "
                    f"yaw={yaw:.3f} deg, pitch={pitch:.3f} deg, "
                    f"roll={roll:.3f} deg"
                )
                state_number += 1

        return "\n".join(lines)
