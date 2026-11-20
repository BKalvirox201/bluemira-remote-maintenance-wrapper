# SPDX-FileCopyrightText: 2026-present N.Salt
#
# SPDX-License-Identifier: MIT 2026
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from operator import itemgetter
from typing import TYPE_CHECKING

import numpy as np
from bluemira.base.look_and_feel import (
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
)
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from bluemira_wrapper.blanket_grouping import group_blanket_components
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.animation import (  # noqa: E501
    PathAnimator,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.constraints import (  # noqa: E501
    PayloadConstraints,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.environment import (  # noqa: E501
    EnvironmentBuilder,
    PayloadBuilder,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.mesh_extraction import (  # noqa: E501
    MeshExtractor,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.models import (
    Mesh,
    Waypoint,
    WaypointRoute,
    WorkspaceBounds,
    euler_degrees_to_rotation_matrix,
    rotation_matrix_to_euler_degrees,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.planning import (  # noqa: E501
    RoutePlanner,
    SinglePathPlanner,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import ArrayLike

    from eudemo.blanket import Blanket
    from eudemo.vacuum_vessel import VacuumVessel


@dataclass(frozen=True)
class RemovalPlan:
    group_key: tuple[str, int]
    create_route: Callable[[Mesh, Sequence[Waypoint] | None], WaypointRoute]


class BlanketRemovalSimulation:
    def __init__(self, blanket: Blanket, vacuum_vessel: VacuumVessel, n_TF: int):
        self.blanket = blanket
        self.vacuum_vessel = vacuum_vessel
        self.n_TF = n_TF

    def run_isolated_removal(self, timeout: int) -> None:
        """Plan isolated removal, calculating a reference point per blanket."""
        bluemira_print("Beginning Blanket Removal Path Planning")
        blanket_meshes = MeshExtractor.blanket_meshes(self.blanket)
        vacuum_vessel_meshes = MeshExtractor.vacuum_vessel_meshes(self.vacuum_vessel)

        sector_walls = EnvironmentBuilder.create_sector_walls(self.n_TF)
        environment = EnvironmentBuilder.create_environment(
            vacuum_vessel_meshes, sector_walls
        )

        animator = PathAnimator()
        workspace_bounds = WorkspaceBounds(-10, 15, 0, 5, -10, 50)

        for blanket_mesh in blanket_meshes:
            blanket_mesh.pivot_point = highest_z_average_xy_point(blanket_mesh.vertices)
            initial_rotation = self._prepare_mesh_for_planning(blanket_mesh)
            bluemira_print(f"Planning {blanket_mesh.name}")
            payload = PayloadBuilder.create_payload(blanket_mesh)

            payload_constraints = PayloadConstraints(0.1, 10, 15, initial_rotation)
            route_factory = (
                self._create_obs_removal_route
                if blanket_mesh.name.startswith("OBS")
                else (
                    self._create_ibs_removal_route
                    if blanket_mesh.name.startswith("IBS")
                    else None
                )
            )
            if route_factory is None:
                raise ValueError(f"Unknown blanket type: {blanket_mesh.name}")
            route = route_factory(blanket_mesh)
            animator.show_waypoints(blanket_mesh, vacuum_vessel_meshes, route)

            single_path_planner = SinglePathPlanner(
                payload,
                blanket_mesh.pivot_point,
                payload_constraints,
                environment,
                workspace_bounds,
                get_centerline_unit_vector(self.n_TF),
            )

            route_planner = RoutePlanner(single_path_planner)
            try:
                planned_route = route_planner.plan(route, timeout)
            except RuntimeError as error:
                bluemira_warn(f"No route found {blanket_mesh.name}: {error}")
                continue

            bluemira_print_flush(planned_route.to_human_readable_string())
            animator.animate(blanket_mesh, vacuum_vessel_meshes, planned_route)

    def _create_group_mesh(
        self,
        blanket_group: list[Mesh],
        pivot_point: ArrayLike | None = None,
    ) -> Mesh:
        vertex_blocks = []
        triangle_blocks = []

        vertex_offset = 0
        for mesh in blanket_group:
            vertex_blocks.append(mesh.vertices)
            triangle_blocks.append(mesh.triangles + vertex_offset)
            vertex_offset += len(mesh.vertices)

        vertices = np.vstack(vertex_blocks)
        pivot_point = _pivot_point(pivot_point, vertices)
        group_mesh = Mesh(
            name=",".join(mesh.name for mesh in blanket_group),
            vertices=vertices,
            triangles=np.vstack(triangle_blocks),
            pivot_point=pivot_point,
        )
        initial_rotation = self._prepare_mesh_for_planning(group_mesh)
        for mesh in blanket_group:
            mesh.vertices = self._rotate_mesh_to_local_frame(
                mesh.vertices, group_mesh.pivot_point, initial_rotation
            )
            mesh.initial_rotation = initial_rotation

        return group_mesh

    def _prepare_mesh_for_planning(self, mesh: Mesh) -> np.ndarray:
        initial_rotation = self._calculate_blanket_world_rotation(mesh)
        mesh.vertices = self._rotate_mesh_to_local_frame(
            mesh.vertices, mesh.pivot_point, initial_rotation
        )
        mesh.initial_rotation = initial_rotation
        return initial_rotation

    @staticmethod
    def _rotate_mesh_to_local_frame(
        vertices: np.ndarray, pivot_point: np.ndarray, world_rotation: np.ndarray
    ) -> np.ndarray:
        return (vertices - pivot_point) @ world_rotation + pivot_point

    @staticmethod
    def _yaw_degrees(rotation_matrix: np.ndarray | None) -> float:
        if rotation_matrix is None:
            return 0.0
        yaw, _, _ = rotation_matrix_to_euler_degrees(rotation_matrix)
        return yaw

    def _create_group_payload(self, blanket_group: list[Mesh]):
        return [PayloadBuilder.create_payload(mesh) for mesh in blanket_group]

    def _create_standard_removal_route(
        self,
        group_mesh: Mesh,
        staging_waypoints: Sequence[Waypoint],
        end_height: float = 18.0,
    ) -> WaypointRoute:
        initial_rotation = (
            np.eye(3)
            if group_mesh.initial_rotation is None
            else group_mesh.initial_rotation
        )
        final_staging_waypoint = staging_waypoints[-1] if staging_waypoints else None
        end_rotation = (
            initial_rotation
            if final_staging_waypoint is None
            else final_staging_waypoint.rotation_matrix
        )
        if end_rotation is None:
            raise RuntimeError("The final staging waypoint has no rotation matrix")
        end_yaw_pitch_roll = rotation_matrix_to_euler_degrees(end_rotation)
        return WaypointRoute(
            waypoints=[
                Waypoint(
                    "starting position",
                    group_mesh.pivot_point,
                    yaw_degrees=self._yaw_degrees(initial_rotation),
                    rotation_matrix=initial_rotation,
                ),
                *staging_waypoints,
                Waypoint(
                    name="end_position",
                    position=(
                        group_mesh.pivot_point
                        if final_staging_waypoint is None
                        else final_staging_waypoint.position
                    )
                    + np.array([0.0, 0.0, end_height]),
                    yaw_degrees=end_yaw_pitch_roll[0],
                    pitch_degrees=end_yaw_pitch_roll[1],
                    roll_degrees=end_yaw_pitch_roll[2],
                    rotation_matrix=end_rotation,
                ),
            ]
        )

    def _create_obs_removal_route(
        self,
        group_mesh: Mesh,
        staging_waypoints: Sequence[Waypoint] | None = None,
        *,
        blanket_groups: Mapping[tuple[str, int], list[Mesh]] | None = None,
        pivot_points_by_group: Mapping[tuple[str, int], ArrayLike] | None = None,
    ) -> WaypointRoute:
        def _calculate_obs_staging_waypoint(
            blanket_mesh: Mesh, position: np.ndarray
        ) -> Waypoint:
            centreline_uv = get_centerline_unit_vector(self.n_TF)
            centreline_angle = np.arctan2(centreline_uv[1], centreline_uv[0])
            normalisation_angle = self._calculate_blanket_yaw_radians(blanket_mesh)
            normalisation = euler_degrees_to_rotation_matrix(
                np.degrees(normalisation_angle), 0.0, 0.0
            )
            perpendicular_uv = np.array([-centreline_uv[1], centreline_uv[0], 0.0])
            pitch_degrees = 0
            pitch_rotation = Rotation.from_rotvec(
                perpendicular_uv * np.radians(pitch_degrees)
            ).as_matrix()
            rotation_matrix = pitch_rotation @ normalisation
            yaw_degrees = float(np.degrees(centreline_angle))
            return Waypoint(
                name="below_upper_port",
                position=position,
                roll_degrees=0.0,
                pitch_degrees=pitch_degrees,
                yaw_degrees=yaw_degrees,
                rotation_matrix=rotation_matrix,
                is_staging=True,
            )

        if staging_waypoints is None:
            if blanket_groups is None:
                middle_positions = [_pivot_point(None, group_mesh.vertices)]
            else:
                obs_groups = sorted(
                    (key for key in blanket_groups if key[0] == "OBS"),
                    key=itemgetter(1),
                )
                if not obs_groups:
                    raise ValueError("At least one OBS blanket group is required")
                middle = len(obs_groups) // 2
                middle_keys = (
                    obs_groups[middle : middle + 1]
                    if len(obs_groups) % 2
                    else obs_groups[middle - 1 : middle + 1]
                )
                middle_positions = [
                    _pivot_point(
                        (
                            None
                            if pivot_points_by_group is None
                            else pivot_points_by_group.get(key)
                        ),
                        np.vstack([mesh.vertices for mesh in blanket_groups[key]]),
                    )
                    for key in middle_keys
                ]
            staging_position = np.mean(middle_positions, axis=0)
            staging_waypoints = [
                _calculate_obs_staging_waypoint(group_mesh, staging_position)
            ]
        return self._create_standard_removal_route(group_mesh, staging_waypoints)

    def _create_ibs_removal_route(
        self,
        group_mesh: Mesh,
        staging_waypoints: Sequence[Waypoint] | None = None,
    ) -> WaypointRoute:
        def _calculate_ibs_staging_waypoint(blanket_mesh: Mesh) -> Waypoint:
            pivot_point = np.asarray(blanket_mesh.pivot_point, dtype=np.float64)
            centreline_uv = get_centerline_unit_vector(self.n_TF)
            centreline_angle = np.arctan2(centreline_uv[1], centreline_uv[0])
            normalisation_angle = self._calculate_blanket_yaw_radians(blanket_mesh)
            normalisation = euler_degrees_to_rotation_matrix(
                np.degrees(normalisation_angle), 0.0, 0.0
            )
            distance_from_origin = 9.6
            pitch_degrees = 0.0
            rotation_matrix = normalisation
            yaw_degrees = float(np.degrees(centreline_angle))
            return Waypoint(
                name="below_upper_port",
                position=np.array([
                    centreline_uv[0] * distance_from_origin,
                    centreline_uv[1] * distance_from_origin,
                    pivot_point[2] - 0.4,
                ]),
                pitch_degrees=pitch_degrees,
                yaw_degrees=yaw_degrees,
                rotation_matrix=rotation_matrix,
                is_staging=True,
            )

        if staging_waypoints is None:
            staging_waypoints = [_calculate_ibs_staging_waypoint(group_mesh)]
        return self._create_standard_removal_route(group_mesh, staging_waypoints)

    def _create_direct_removal_route(
        self,
        group_mesh: Mesh,
        staging_waypoints: Sequence[Waypoint] | None = None,
    ) -> WaypointRoute:
        return self._create_standard_removal_route(
            group_mesh, [] if staging_waypoints is None else staging_waypoints
        )

    def _calculate_blanket_yaw_radians(self, blanket_mesh: Mesh) -> float:
        vertices = points_from_vertices(blanket_mesh.vertices)
        hull = ConvexHull(vertices[:, :2])
        rectangle = Polygon(vertices[hull.vertices, :2]).minimum_rotated_rectangle
        rectangle_points = np.asarray(rectangle.boundary.coords[:-1])
        edge_vectors = np.diff(
            np.vstack([rectangle_points, rectangle_points[0]]), axis=0
        )
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        longest_edge = edge_vectors[np.argmax(edge_lengths)]
        centreline_uv = get_centerline_unit_vector(self.n_TF)
        tangent_uv = np.array([-centreline_uv[1], centreline_uv[0]])
        centreline_alignment = np.dot(longest_edge, centreline_uv[:2])
        tangent_alignment = np.dot(longest_edge, tangent_uv)
        reference_alignment = (
            centreline_alignment
            if abs(centreline_alignment) >= abs(tangent_alignment)
            else tangent_alignment
        )
        if reference_alignment < 0:
            longest_edge = -longest_edge
        blanket_angle = np.arctan2(longest_edge[1], longest_edge[0])
        centreline_angle = np.arctan2(centreline_uv[1], centreline_uv[0])
        return float(
            np.arctan2(
                np.sin(centreline_angle - blanket_angle),
                np.cos(centreline_angle - blanket_angle),
            )
        )

    def _calculate_blanket_world_rotation(self, _blanket_mesh: Mesh) -> np.ndarray:
        sector_bisector_degrees = 180.0 / self.n_TF
        return euler_degrees_to_rotation_matrix(sector_bisector_degrees, 0.0, 0.0)

    def run_blanket_sequential_removal(
        self,
        timeout: int,
        staging_waypoints_by_group: (
            Mapping[tuple[str, int], Sequence[Waypoint]] | None
        ) = None,
        pivot_points_by_group: Mapping[tuple[str, int], ArrayLike] | None = None,
    ) -> None:
        """Plan sequential removal with per-group waypoint/reference overrides.

        Waypoints in each sequence are visited in order. An empty sequence
        disables staging for that group; groups omitted from the mapping use
        their default staging waypoint.
        Pivot points omitted from the mapping use a calculated group pivot.
        """
        animator = PathAnimator()

        bluemira_print("Beginning Blanket Sequential Removal")
        blanket_meshes = MeshExtractor.blanket_meshes(self.blanket)
        vacuum_vessel_meshes = MeshExtractor.vacuum_vessel_meshes(self.vacuum_vessel)
        sector_walls = EnvironmentBuilder.create_sector_walls(self.n_TF)
        blanket_groups = group_blanket_components(blanket_meshes)
        removal_order = [
            RemovalPlan(("OBS", 1), self._create_direct_removal_route),
            RemovalPlan(
                ("OBS", 0),
                partial(
                    self._create_obs_removal_route,
                    blanket_groups=blanket_groups,
                    pivot_points_by_group=pivot_points_by_group,
                ),
            ),
            RemovalPlan(
                ("OBS", 2),
                partial(
                    self._create_obs_removal_route,
                    blanket_groups=blanket_groups,
                    pivot_points_by_group=pivot_points_by_group,
                ),
            ),
            RemovalPlan(("IBS", 0), self._create_ibs_removal_route),
            RemovalPlan(("IBS", 1), self._create_ibs_removal_route),
        ]

        unknown_groups = [
            plan.group_key
            for plan in removal_order
            if plan.group_key not in blanket_groups
        ]
        if unknown_groups:
            raise ValueError(f"Unknown blanket groups: {unknown_groups}")

        unknown_staging_groups = (
            set(staging_waypoints_by_group or {}) - blanket_groups.keys()
        )
        if unknown_staging_groups:
            raise ValueError(
                f"Unknown staging waypoint groups: {sorted(unknown_staging_groups)}"
            )
        unknown_pivot_groups = set(pivot_points_by_group or {}) - blanket_groups.keys()
        if unknown_pivot_groups:
            raise ValueError(
                f"Unknown blanket pivot point groups: {sorted(unknown_pivot_groups)}"
            )
        remaining_blankets = list(blanket_meshes)
        workspace_bounds = WorkspaceBounds(-10, 15, 0, 5, -10, 50)

        for removal_plan in removal_order:
            group_key = removal_plan.group_key
            blanket_group = blanket_groups[group_key]
            default_pivot_point = (
                ibs_group_pivot_point(blanket_group) if group_key[0] == "IBS" else None
            )
            group_mesh = self._create_group_mesh(
                blanket_group,
                (
                    pivot_points_by_group.get(group_key, default_pivot_point)
                    if pivot_points_by_group is not None
                    else default_pivot_point
                ),
            )
            bluemira_print(f"Planning group {group_key}: {group_mesh.name}")
            group_names = {mesh.name for mesh in blanket_group}

            environment_blankets = [
                mesh for mesh in remaining_blankets if mesh.name not in group_names
            ]
            remaining_blankets = environment_blankets
            environment_meshes = vacuum_vessel_meshes + environment_blankets
            environment = EnvironmentBuilder.create_environment(
                environment_meshes, sector_walls
            )

            payload = self._create_group_payload(blanket_group)
            route = removal_plan.create_route(
                group_mesh,
                (
                    None
                    if staging_waypoints_by_group is None
                    else staging_waypoints_by_group.get(group_key)
                ),
            )
            animator.show_waypoints(group_mesh, environment_meshes, route)

            payload_constraints = PayloadConstraints(
                3, 3, 20, group_mesh.initial_rotation
            )
            single_path_planner = SinglePathPlanner(
                payload,
                group_mesh.pivot_point,
                payload_constraints,
                environment,
                workspace_bounds,
                get_centerline_unit_vector(self.n_TF),
            )
            route_planner = RoutePlanner(single_path_planner)

            try:
                planned_route = route_planner.plan(route, timeout)

            except RuntimeError as error:
                bluemira_warn(f"No route found {group_mesh.name}: {error}")
                continue

            else:
                bluemira_print_flush(planned_route.to_human_readable_string())
                animator.animate(group_mesh, environment_meshes, planned_route)


def _pivot_point(pivot_point: ArrayLike | None, vertices: np.ndarray) -> np.ndarray:
    if pivot_point is None:
        return highest_z_average_xy_point(vertices)

    point = np.asarray(pivot_point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("A blanket pivot point must contain three finite coordinates")
    return point.copy()


def highest_z_average_xy_point(vertices: np.ndarray) -> np.ndarray:
    """Return mean X/Y and maximum Z coordinates for blanket vertices."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Blanket vertices must be a non-empty array of 3D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Blanket vertices must contain only finite coordinates")
    highest_z_points = points[points[:, 2] == points[:, 2].max()]
    return np.array(
        [
            highest_z_points[:, 0].mean(),
            highest_z_points[:, 1].mean(),
            highest_z_points[0, 2],
        ],
        dtype=np.float64,
    )


def ibs_group_pivot_point(blanket_group: Sequence[Mesh]) -> np.ndarray:
    """Return the X centre, mean Y, and maximum Z for an IBS group."""
    points = np.vstack([mesh.vertices for mesh in blanket_group]).astype(
        np.float64, copy=False
    )
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Blanket vertices must be a non-empty array of 3D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("Blanket vertices must contain only finite coordinates")
    return np.array(
        [
            (points[:, 0].min() + points[:, 0].max()) / 2.0,
            points[:, 1].mean(),
            points[:, 2].max(),
        ],
        dtype=np.float64,
    )


def get_centerline_unit_vector(n_TF: int):
    bisector_angle = 180.0 / n_TF
    centerline_direction = np.array([
        np.cos(np.radians(bisector_angle)),
        np.sin(np.radians(bisector_angle)),
        0.0,
    ])
    centerline_direction /= np.linalg.norm(centerline_direction)
    return centerline_direction


def points_from_vertices(vertices) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[1] == 3:
        return points
    return points.T
