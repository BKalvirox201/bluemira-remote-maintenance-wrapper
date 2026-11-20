from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np
from scipy.spatial.transform import Rotation

from .models import rotation_matrix_to_euler_degrees


@dataclass
class PayloadConstraints:
    maximum_roll_degrees: float | None = None
    maximum_pitch_degrees: float | None = None
    maximum_yaw_degrees: float | None = None
    reference_rotation: np.ndarray | None = None

    def valid_orientation(self, rotation_matrix: np.ndarray) -> bool:
        reference_rotation = (
            np.eye(3) if self.reference_rotation is None else self.reference_rotation
        )
        relative_rotation = rotation_matrix @ reference_rotation.T
        yaw, pitch, roll = rotation_matrix_to_euler_degrees(relative_rotation)

        if (
            self.maximum_roll_degrees is not None
            and abs(roll) > self.maximum_roll_degrees
        ):
            return False
        if (
            self.maximum_pitch_degrees is not None
            and abs(pitch) > self.maximum_pitch_degrees
        ):
            return False
        if self.maximum_yaw_degrees is not None and abs(yaw) > self.maximum_yaw_degrees:
            return False

        return True


class PayloadStateValidator:
    def __init__(
        self,
        payload: list[fcl.CollisionObject],
        environment: list[tuple[str, np.ndarray, fcl.CollisionObject]],
        constraints: PayloadConstraints,
        pivot_point: np.ndarray,
    ):
        self.payload = payload
        self.environment = environment
        self.constraints = constraints
        self.pivot_point = np.asarray(pivot_point, dtype=float)
        self.request = fcl.CollisionRequest()
        self.diagnostic_waypoint_name: str | None = None
        self.last_invalid_reason: str | None = None

    @staticmethod
    def state_to_rotation_matrix(state) -> np.ndarray:
        quaternion = state.rotation()
        return Rotation.from_quat([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]).as_matrix()

    def __call__(self, state) -> bool:
        self.last_invalid_reason = None
        world_position = np.array(
            [state.getX(), state.getY(), state.getZ()], dtype=float
        )

        rotation_matrix = self.state_to_rotation_matrix(state)
        translation = world_position - rotation_matrix @ self.pivot_point

        reference_rotation = self.constraints.reference_rotation
        is_start_pose = np.allclose(world_position, self.pivot_point) and np.allclose(
            rotation_matrix,
            reference_rotation if reference_rotation is not None else np.eye(3),
        )

        if not self.constraints.valid_orientation(rotation_matrix):
            self.last_invalid_reason = "orientation constraint"
            return False

        transform = fcl.Transform(rotation_matrix, translation)
        for payload_object in self.payload:
            payload_object.setTransform(transform)

        colliding_mesh_names = set()
        for payload_object in self.payload:
            for mesh_name, _, environment_object in self.environment:
                collision_result = fcl.CollisionResult()
                if (
                    fcl.collide(
                        payload_object,
                        environment_object,
                        self.request,
                        collision_result,
                    )
                    > 0
                ):
                    colliding_mesh_names.add(mesh_name)
                    if not is_start_pose and self.diagnostic_waypoint_name is None:
                        return False

        if colliding_mesh_names:
            self.last_invalid_reason = f"collision with {sorted(colliding_mesh_names)}"

        return not colliding_mesh_names
