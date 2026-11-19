from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bluemira_wrapper.fcl_ompl_path_planner import (
    blanket_removal_simulation as simulation_module,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_removal_simulation import (
    BlanketRemovalSimulation,
    get_centerline_unit_vector,
)
from bluemira_wrapper.fcl_ompl_path_planner.lib import (
    constraints as constraints_module,
)
from bluemira_wrapper.fcl_ompl_path_planner.lib import (
    mesh_extraction as mesh_extraction_module,
)
from bluemira_wrapper.fcl_ompl_path_planner.lib.animation import PathAnimator
from bluemira_wrapper.fcl_ompl_path_planner.lib.constraints import (
    PayloadConstraints,
    PayloadStateValidator,
)
from bluemira_wrapper.fcl_ompl_path_planner.lib.environment import EnvironmentBuilder
from bluemira_wrapper.fcl_ompl_path_planner.lib.mesh_extraction import MeshExtractor
from bluemira_wrapper.fcl_ompl_path_planner.lib.models import (
    Mesh,
    Waypoint,
    WorkspaceBounds,
    euler_degrees_to_rotation_matrix,
    rotation_matrix_to_euler_degrees,
)
from bluemira_wrapper.fcl_ompl_path_planner.lib.planning import SinglePathPlanner


class FakeCollisionObject:
    def __init__(self, name="payload"):
        self.name = name
        self.transform = None

    def setTransform(self, transform):
        self.transform = transform


def _make_state(position, rotation_matrix):
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    rotation = SimpleNamespace(
        x=quaternion[0],
        y=quaternion[1],
        z=quaternion[2],
        w=quaternion[3],
    )
    return SimpleNamespace(
        getX=lambda: float(position[0]),
        getY=lambda: float(position[1]),
        getZ=lambda: float(position[2]),
        rotation=lambda: rotation,
    )


def _make_mesh(name, vertices, triangles=None):
    vertices = np.asarray(vertices, dtype=float)
    return Mesh(
        name=name,
        vertices=vertices,
        triangles=(
            np.empty((0, 3), dtype=np.int32)
            if triangles is None
            else np.asarray(triangles, dtype=np.int32)
        ),
        pivot_point=vertices.mean(axis=0),
    )


def test_group_mesh_rotation_pivot_is_vertex_weighted():
    first = Mesh(
        name="IBS_FW_0 1",
        vertices=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        pivot_point=np.array([2.0 / 3, 2.0 / 3, 0.0]),
    )
    second = Mesh(
        name="IBS_BZ_0 1",
        vertices=np.array([[4.0, 0.0, 0.0], [4.0, 2.0, 0.0]]),
        triangles=np.empty((0, 3), dtype=np.int32),
        pivot_point=np.array([4.0, 1.0, 0.0]),
    )
    simulation = BlanketRemovalSimulation(None, None, n_TF=24)
    original_vertices = [first.vertices.copy(), second.vertices.copy()]

    all_vertices = np.vstack(original_vertices)
    expected_pivot_point = np.array(
        [
            all_vertices[:, 0].mean(),
            all_vertices[:, 1].mean(),
            all_vertices[:, 2].max(),
        ]
    )
    grouped = simulation._create_group_mesh([first, second])

    np.testing.assert_allclose(grouped.pivot_point, expected_pivot_point)
    for mesh, vertices in zip([first, second], original_vertices, strict=True):
        reconstructed_vertices = (
            mesh.vertices - grouped.pivot_point
        ) @ grouped.initial_rotation.T + grouped.pivot_point
        np.testing.assert_allclose(reconstructed_vertices, vertices, atol=1e-12)


def test_group_mesh_accepts_a_custom_pivot_point():
    mesh = _make_mesh(
        "IBS_FW_0 1",
        [[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 3.0]],
    )
    pivot_point = np.array([4.0, 5.0, 6.0])
    original_vertices = mesh.vertices.copy()
    simulation = BlanketRemovalSimulation(None, None, n_TF=24)

    grouped = simulation._create_group_mesh([mesh], pivot_point)

    np.testing.assert_array_equal(grouped.pivot_point, pivot_point)
    restored_vertices = (
        grouped.vertices - grouped.pivot_point
    ) @ grouped.initial_rotation.T + grouped.pivot_point
    np.testing.assert_allclose(restored_vertices, original_vertices, atol=1e-12)
    np.testing.assert_allclose(
        simulation._rotate_mesh_to_local_frame(
            pivot_point[None, :], pivot_point, grouped.initial_rotation
        )[0],
        pivot_point,
    )


def test_default_pivot_uses_average_xy_and_highest_z():
    vertices = np.array(
        [
            [100.0, 200.0, 1.0],
            [4.0, 6.0, 5.0],
            [2.0, 4.0, 3.0],
            [8.0, 10.0, 5.0],
        ]
    )

    point = simulation_module.highest_z_average_xy_point(vertices)

    np.testing.assert_array_equal(point, [6.0, 8.0, 5.0])


def test_ibs_pivot_point_centres_x_and_averages_y():
    blanket_group = [
        _make_mesh(
            "IBS_FW_0 1",
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 3.0],
                [10.0, 10.0, 3.0],
                [4.0, 10.0, 0.0],
            ],
        )
    ]

    point = simulation_module.ibs_group_pivot_point(blanket_group)

    np.testing.assert_array_equal(point, [5.0, 5.0, 3.0])


@pytest.mark.parametrize(
    ("group_indices", "expected"),
    [
        ([0, 1, 2], [12.0, 1.0, 2.0]),
        ([0, 1, 2, 3], [17.0, 1.0, 2.0]),
    ],
)
def test_obs_route_uses_middle_group_start_position_for_staging(
    group_indices, expected
):
    blanket_groups = {
        ("OBS", index): [
            _make_mesh(
                f"OBS_FW_{index} 1",
                [
                    [float(index * 10), 0.0, 0.0],
                    [float(index * 10 + 2), 0.0, 2.0],
                    [float(index * 10 + 2), 2.0, 2.0],
                    [float(index * 10), 2.0, 0.0],
                ],
            )
        ]
        for index in reversed(group_indices)
    }
    simulation = BlanketRemovalSimulation(None, None, n_TF=24)

    route = simulation._create_obs_removal_route(
        blanket_groups[("OBS", group_indices[0])][0],
        blanket_groups=blanket_groups,
    )

    np.testing.assert_array_equal(route.waypoints[1].position, expected)


def test_start_rotation_uses_sector_bisector_and_restores_mesh_orientation():
    n_tf = 24
    bisector_angle = np.deg2rad(180.0 / n_tf)
    tangent_angle = bisector_angle + np.pi / 2
    rotation = np.array([
        [np.cos(tangent_angle), -np.sin(tangent_angle)],
        [np.sin(tangent_angle), np.cos(tangent_angle)],
    ])
    footprint = (
        np.array([
            [-2.0, -0.5],
            [2.0, -0.5],
            [2.0, 0.5],
            [-2.0, 0.5],
        ])
        @ rotation.T
    )
    vertices = np.column_stack([footprint, np.zeros(len(footprint))])
    blanket_mesh = Mesh(
        name="OBS_FW_0 1",
        vertices=vertices,
        triangles=np.empty((0, 3), dtype=np.int32),
        pivot_point=vertices.mean(axis=0),
    )
    simulation = BlanketRemovalSimulation(None, None, n_tf)

    original_vertices = blanket_mesh.vertices.copy()
    grouped_mesh = simulation._create_group_mesh([blanket_mesh])
    route = simulation._create_standard_removal_route(
        grouped_mesh,
        [
            Waypoint(
                "below_upper_port",
                np.zeros(3),
                rotation_matrix=np.eye(3),
                is_staging=True,
            )
        ],
    )
    initial_yaw = rotation_matrix_to_euler_degrees(route.start.rotation_matrix)[0]
    reconstructed_vertices = (
        grouped_mesh.vertices - grouped_mesh.pivot_point
    ) @ route.start.rotation_matrix.T + grouped_mesh.pivot_point

    np.testing.assert_allclose(
        initial_yaw,
        180.0 / n_tf,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        get_centerline_unit_vector(n_tf)[:2],
        [np.cos(bisector_angle), np.sin(bisector_angle)],
    )
    np.testing.assert_allclose(reconstructed_vertices, original_vertices, atol=1e-12)


def test_environment_keeps_collision_meshes_separate(monkeypatch):
    vessel_mesh = Mesh(
        name="vacuum_vessel",
        vertices=np.array([[1.0, 2.0, 3.0]]),
        triangles=np.empty((0, 3), dtype=np.int32),
        pivot_point=np.array([1.0, 2.0, 3.0]),
    )
    wall_mesh = Mesh(
        name="sector_wall",
        vertices=np.array([[4.0, 5.0, 6.0]]),
        triangles=np.empty((0, 3), dtype=np.int32),
        pivot_point=np.array([4.0, 5.0, 6.0]),
    )
    monkeypatch.setattr(
        EnvironmentBuilder,
        "create_collision_object",
        staticmethod(lambda meshes: meshes[0].name),
    )

    environment = EnvironmentBuilder.create_environment([vessel_mesh], [wall_mesh])

    np.testing.assert_equal(
        [name for name, _, _ in environment], ["vacuum_vessel", "sector_wall"]
    )
    np.testing.assert_array_equal(environment[0][1], vessel_mesh.pivot_point)
    np.testing.assert_array_equal(environment[1][1], wall_mesh.pivot_point)
    np.testing.assert_equal(
        [collision_object for _, _, collision_object in environment],
        ["vacuum_vessel", "sector_wall"],
    )


def test_shape_mesh_extraction_preserves_facet_coordinates(monkeypatch):
    facet_points = [
        (10.0, 2.0, 0.0),
        (9.0, 3.0, 1.0),
        (8.0, 4.0, 2.0),
    ]
    monkeypatch.setattr(
        mesh_extraction_module.MeshPart,
        "meshFromShape",
        lambda **_: SimpleNamespace(Facets=[SimpleNamespace(Points=facet_points)]),
    )
    source_shape = object()

    extracted = MeshExtractor.shape_to_mesh(source_shape, "sector_mesh")

    np.testing.assert_array_equal(extracted.vertices, facet_points)
    np.testing.assert_array_equal(extracted.triangles, [[0, 1, 2]])
    np.testing.assert_allclose(extracted.pivot_point, np.mean(facet_points, axis=0))


@pytest.mark.parametrize("n_tf", [16, 24, 32])
def test_world_rotation_is_sector_bisector(n_tf):
    simulation = BlanketRemovalSimulation(None, None, n_tf)

    yaw, pitch, roll = rotation_matrix_to_euler_degrees(
        simulation._calculate_blanket_world_rotation(None)
    )

    np.testing.assert_allclose([yaw, pitch, roll], [180.0 / n_tf, 0.0, 0.0])


@pytest.mark.parametrize(
    ("yaw_degrees", "expected_valid"),
    [(11.25, True), (19.2, True), (19.3, False)],
)
def test_orientation_limits_are_relative_to_start_rotation(yaw_degrees, expected_valid):
    reference_rotation = euler_degrees_to_rotation_matrix(11.25, 0.0, 0.0)
    candidate_rotation = euler_degrees_to_rotation_matrix(yaw_degrees, 0.0, 0.0)
    constraints = PayloadConstraints(
        maximum_yaw_degrees=8.0,
        reference_rotation=reference_rotation,
    )

    np.testing.assert_equal(
        constraints.valid_orientation(candidate_rotation), expected_valid
    )


def test_start_pose_restores_payload_to_world_pivot_point(monkeypatch, capsys):
    pivot_point = np.array([2.0, 1.0, -1.0])
    world_position = np.array([4.0, 5.0, 6.0])
    rotation = euler_degrees_to_rotation_matrix(90.0, 0.0, 0.0)
    payload_object = FakeCollisionObject()
    validator = PayloadStateValidator(
        [payload_object],
        [],
        PayloadConstraints(reference_rotation=rotation),
        pivot_point,
    )

    def transform(matrix, translation):
        return matrix.copy(), translation.copy()

    monkeypatch.setattr(constraints_module.fcl, "Transform", transform)
    validator.diagnostic_waypoint_name = "test pose"

    np.testing.assert_equal(validator(_make_state(world_position, rotation)), True)

    actual_rotation, actual_translation = payload_object.transform
    np.testing.assert_allclose(actual_rotation, rotation)
    np.testing.assert_allclose(
        actual_translation, world_position - rotation @ pivot_point
    )
    np.testing.assert_allclose(
        actual_rotation @ pivot_point + actual_translation, world_position
    )
    np.testing.assert_equal(capsys.readouterr().out, "")


def test_pose_matrices_place_pivot_at_waypoint_position():
    pivot_point = np.array([2.0, 1.0, -1.0])
    waypoint_position = np.array([4.0, 5.0, 6.0])
    rotation = euler_degrees_to_rotation_matrix(90.0, 0.0, 0.0)
    state = _make_state(waypoint_position, rotation)

    planner_pose = SinglePathPlanner.pose_matrix(state, pivot_point)
    animation_pose = PathAnimator.create_pose(
        waypoint_position, rotation, pivot_point
    )

    for pose in (planner_pose, animation_pose):
        np.testing.assert_allclose(
            pose[:3, :3] @ pivot_point + pose[:3, 3], waypoint_position
        )


def test_start_pose_reports_named_environment_collision(monkeypatch, capsys):
    pivot_point = np.array([10.0, 2.0, 0.0])
    payload_object = FakeCollisionObject()
    colliding_obstacle = FakeCollisionObject("sector_wall_2")
    environment = [
        ("vacuum_vessel", np.array([8.0, 1.0, 0.0]), FakeCollisionObject()),
        ("sector_wall_2", np.array([9.0, 3.0, 0.0]), colliding_obstacle),
    ]
    validator = PayloadStateValidator(
        [payload_object],
        environment,
        PayloadConstraints(reference_rotation=np.eye(3)),
        pivot_point,
    )
    monkeypatch.setattr(
        constraints_module.fcl,
        "collide",
        lambda payload, obstacle, request, result: int(obstacle is colliding_obstacle),
    )

    np.testing.assert_equal(validator(_make_state(pivot_point, np.eye(3))), False)

    np.testing.assert_equal(
        validator.last_invalid_reason, "collision with ['sector_wall_2']"
    )
    np.testing.assert_equal(capsys.readouterr().out, "")


def test_staging_pose_reports_orientation_rejection(monkeypatch, capsys):
    reference_rotation = euler_degrees_to_rotation_matrix(11.25, 0.0, 0.0)
    invalid_rotation = euler_degrees_to_rotation_matrix(20.0, 0.0, 0.0)
    validator = PayloadStateValidator(
        [FakeCollisionObject()],
        [],
        PayloadConstraints(
            maximum_yaw_degrees=8.0,
            reference_rotation=reference_rotation,
        ),
        np.zeros(3),
    )
    validator.diagnostic_waypoint_name = "below_upper_port"

    np.testing.assert_equal(
        validator(_make_state([1.0, 2.0, 3.0], invalid_rotation)), False
    )

    np.testing.assert_equal(validator.last_invalid_reason, "orientation constraint")
    np.testing.assert_equal(capsys.readouterr().out, "")


def test_staging_pose_reports_collision_with_environment_mesh(monkeypatch, capsys):
    payload_object = FakeCollisionObject()
    obstacle = FakeCollisionObject("OBS_FW_0 1")
    validator = PayloadStateValidator(
        [payload_object],
        [("OBS_FW_0 1", np.array([10.0, 2.0, 0.0]), obstacle)],
        PayloadConstraints(reference_rotation=np.eye(3)),
        np.zeros(3),
    )
    validator.diagnostic_waypoint_name = "below_upper_port"
    monkeypatch.setattr(
        constraints_module.fcl,
        "collide",
        lambda payload, environment, request, result: 1,
    )

    np.testing.assert_equal(
        validator(
            _make_state([1.0, 2.0, 3.0], np.eye(3)),
        ),
        False,
    )

    np.testing.assert_equal(
        validator.last_invalid_reason, "collision with ['OBS_FW_0 1']"
    )
    np.testing.assert_equal(capsys.readouterr().out, "")


def test_planner_includes_validator_reason_in_waypoint_error(monkeypatch):
    planner = SinglePathPlanner(
        [],
        np.zeros(3),
        PayloadConstraints(),
        [],
        WorkspaceBounds(-1, 1, -1, 1, -1, 1),
        np.array([1.0, 0.0, 0.0]),
    )
    validator = PayloadStateValidator([], [], PayloadConstraints(), np.zeros(3))
    validator.last_invalid_reason = "collision with ['upper_port']"
    monkeypatch.setattr(planner, "create_validator", lambda: validator)
    monkeypatch.setattr(
        planner,
        "create_space_information",
        lambda _: (None, SimpleNamespace(isValid=lambda _: False)),
    )
    monkeypatch.setattr(planner, "create_state", lambda *_: object())

    with pytest.raises(
        RuntimeError,
        match=r"Invalid waypoint: below_upper_port \(collision with \['upper_port'\]\)",
    ):
        planner.validate(
            Waypoint("below_upper_port", np.ones(3), rotation_matrix=np.eye(3))
        )


def test_standard_route_preserves_start_and_staging_poses():
    pivot_point = np.array([3.0, 4.0, 5.0])
    initial_rotation = euler_degrees_to_rotation_matrix(11.25, 0.0, 0.0)
    staging_rotation = euler_degrees_to_rotation_matrix(18.0, 7.0, 0.0)
    group_mesh = _make_mesh(
        "OBS_FW_1 1",
        [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
    )
    group_mesh.pivot_point = pivot_point
    group_mesh.initial_rotation = initial_rotation
    staging_waypoint = Waypoint(
        name="below_upper_port",
        position=np.array([7.0, 8.0, 9.0]),
        pitch_degrees=7.0,
        yaw_degrees=18.0,
        rotation_matrix=staging_rotation,
        is_staging=True,
    )
    simulation = BlanketRemovalSimulation(None, None, n_TF=16)

    route = simulation._create_standard_removal_route(group_mesh, [staging_waypoint])

    np.testing.assert_equal(route.start.name, "starting position")
    np.testing.assert_equal(route.waypoints[1].name, "below_upper_port")
    np.testing.assert_equal(route.waypoints[1].is_staging, True)
    np.testing.assert_array_equal(route.start.position, pivot_point)
    np.testing.assert_allclose(route.start.rotation_matrix, initial_rotation)
    np.testing.assert_array_equal(route.waypoints[1].position, [7.0, 8.0, 9.0])
    np.testing.assert_allclose(route.waypoints[1].rotation_matrix, staging_rotation)
    np.testing.assert_array_equal(route.waypoints[2].position, [7.0, 8.0, 29.0])
    np.testing.assert_allclose(route.waypoints[2].rotation_matrix, staging_rotation)


def test_standard_route_visits_multiple_staging_waypoints_in_order():
    group_mesh = _make_mesh("OBS_FW_1 1", [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    group_mesh.initial_rotation = euler_degrees_to_rotation_matrix(11.25, 0.0, 0.0)
    first_staging = Waypoint(
        "first_staging",
        np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
        is_staging=True,
    )
    second_staging = Waypoint(
        "second_staging",
        np.array([4.0, 5.0, 6.0]),
        rotation_matrix=euler_degrees_to_rotation_matrix(18.0, 7.0, 0.0),
        is_staging=True,
    )
    simulation = BlanketRemovalSimulation(None, None, n_TF=16)

    route = simulation._create_standard_removal_route(
        group_mesh, [first_staging, second_staging]
    )

    np.testing.assert_equal(
        [waypoint.name for waypoint in route.waypoints],
        ["starting position", "first_staging", "second_staging", "end_position"],
    )
    np.testing.assert_array_equal(
        route.goal.position, second_staging.position + [0.0, 0.0, 20.0]
    )
    np.testing.assert_allclose(
        route.goal.rotation_matrix, second_staging.rotation_matrix
    )


def test_standard_route_without_staging_goes_directly_to_end():
    group_mesh = _make_mesh("OBS_FW_1 1", [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    group_mesh.initial_rotation = euler_degrees_to_rotation_matrix(11.25, 0.0, 0.0)
    simulation = BlanketRemovalSimulation(None, None, n_TF=16)

    route = simulation._create_standard_removal_route(group_mesh, [])

    np.testing.assert_equal(
        [waypoint.name for waypoint in route.waypoints],
        ["starting position", "end_position"],
    )
    np.testing.assert_array_equal(
        route.goal.position, group_mesh.pivot_point + [0.0, 0.0, 20.0]
    )
    np.testing.assert_allclose(route.goal.rotation_matrix, group_mesh.initial_rotation)


def test_sequential_removal_assigns_staging_waypoints_per_group(monkeypatch):
    blanket_meshes = [
        _make_mesh(
            f"{blanket_type}_FW_{index} 1",
            (
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 3.0],
                    [10.0, 10.0, 3.0],
                    [0.0, 10.0, 0.0],
                ]
                if blanket_type == "IBS" and index == 0
                else [
                    [1.0 + index * 10, 2.0, 3.0],
                    [2.0 + index * 10, 2.0, 3.0],
                    [2.0 + index * 10, 3.0, 3.0],
                    [1.0 + index * 10, 3.0, 3.0],
                ]
            ),
        )
        for blanket_type, index in [
            ("OBS", 1),
            ("OBS", 0),
            ("OBS", 2),
            ("IBS", 0),
            ("IBS", 1),
        ]
    ]
    planned_routes = []

    class FakeAnimator:
        def show_waypoints(self, *_):
            pass

        def animate(self, *_):
            pass

    class FakeRoutePlanner:
        def __init__(self, _):
            pass

        def plan(self, route, _):
            planned_routes.append(route)
            return SimpleNamespace(to_human_readable_string=lambda: "planned")

    monkeypatch.setattr(
        simulation_module.MeshExtractor,
        "blanket_meshes",
        staticmethod(lambda _: blanket_meshes),
    )
    monkeypatch.setattr(
        simulation_module.MeshExtractor,
        "vacuum_vessel_meshes",
        staticmethod(lambda _: []),
    )
    monkeypatch.setattr(
        simulation_module.EnvironmentBuilder,
        "create_sector_walls",
        staticmethod(lambda _: []),
    )
    monkeypatch.setattr(
        simulation_module.EnvironmentBuilder,
        "create_environment",
        staticmethod(lambda *_: []),
    )
    monkeypatch.setattr(simulation_module, "PathAnimator", FakeAnimator)
    monkeypatch.setattr(
        simulation_module.PayloadBuilder,
        "create_payload",
        staticmethod(lambda _: FakeCollisionObject()),
    )
    monkeypatch.setattr(simulation_module, "SinglePathPlanner", lambda *_: object())
    monkeypatch.setattr(simulation_module, "RoutePlanner", FakeRoutePlanner)
    simulation = BlanketRemovalSimulation(None, None, n_TF=16)
    first_staging = Waypoint("first", np.array([1.0, 2.0, 3.0]), is_staging=True)
    second_staging = Waypoint("second", np.array([4.0, 5.0, 6.0]), is_staging=True)

    simulation.run_blanket_sequential_removal(
        timeout=1,
        staging_waypoints_by_group={
            ("OBS", 1): [],
            ("OBS", 0): [first_staging, second_staging],
        },
    )

    np.testing.assert_equal(
        [
            [waypoint.name for waypoint in route.waypoints]
            for route in planned_routes[:2]
        ],
        [
            ["starting position", "end_position"],
            ["starting position", "first", "second", "end_position"],
        ],
    )
    np.testing.assert_equal(planned_routes[2].waypoints[1].name, "below_upper_port")
    np.testing.assert_array_equal(
        planned_routes[2].waypoints[1].position, [11.5, 2.5, 3.0]
    )
    np.testing.assert_array_equal(planned_routes[3].start.position, [5.0, 5.0, 3.0])


@pytest.mark.parametrize(
    "blanket_type",
    ["OBS", "IBS"],
)
def test_blanket_removal_routes_create_staging_waypoints(blanket_type):
    blanket_mesh = _make_mesh(
        f"{blanket_type}_FW_0 1",
        [[9.0, 1.0, 1.0], [11.0, 1.0, 1.0], [11.0, 3.0, 3.0], [9.0, 3.0, 3.0]],
    )
    simulation = BlanketRemovalSimulation(None, None, n_TF=16)
    create_route = (
        simulation._create_obs_removal_route
        if blanket_type == "OBS"
        else simulation._create_ibs_removal_route
    )

    route = create_route(blanket_mesh)

    staging_waypoint = route.waypoints[1]
    np.testing.assert_equal(staging_waypoint.is_staging, True)
    if blanket_type == "OBS":
        expected_position = simulation_module.highest_z_average_xy_point(
            blanket_mesh.vertices
        )
    else:
        expected_position = np.array(
            [
                *(get_centerline_unit_vector(16)[:2] * 9.6),
                blanket_mesh.pivot_point[2] - 0.4,
            ]
        )
    np.testing.assert_allclose(
        staging_waypoint.position,
        expected_position,
    )


def test_isolated_removal_uses_existing_routes_and_continues_after_failure(
    monkeypatch, caplog
):
    blanket_meshes = [
        _make_mesh(
            "OBS_FW_0 1",
            [[9.0, 1.0, 0.0], [11.0, 1.0, 0.0], [11.0, 3.0, 2.0], [9.0, 3.0, 2.0]],
            [[0, 1, 2], [0, 2, 3]],
        ),
        _make_mesh(
            "IBS_FW_0 1",
            [[4.0, 0.0, 2.0], [6.0, 0.0, 2.0], [6.0, 2.0, 4.0], [4.0, 2.0, 4.0]],
            [[0, 1, 2], [0, 2, 3]],
        ),
    ]
    vacuum_vessel_meshes = [_make_mesh("vacuum_vessel", [[1, 1, 1], [2, 2, 2]])]
    sector_walls = [_make_mesh("sector_wall", [[0, 0, 0], [1, 1, 1]])]
    environment = [("vacuum_vessel", np.ones(3), FakeCollisionObject())]
    animator_calls = {"waypoints": [], "animations": []}
    planned_inputs = []
    route_calls = []
    payloads = []

    class FakePlannedRoute:
        def to_human_readable_string(self):
            return "planned"

    sentinel_plan = FakePlannedRoute()

    class FakeAnimator:
        def show_waypoints(self, payload_mesh, environment_meshes, route):
            animator_calls["waypoints"].append((
                payload_mesh.name,
                [mesh.name for mesh in environment_meshes],
                route,
            ))

        def animate(self, payload_mesh, environment_meshes, planned_route):
            animator_calls["animations"].append((
                payload_mesh.name,
                [mesh.name for mesh in environment_meshes],
                planned_route,
            ))

    class FakeSinglePathPlanner:
        def __init__(self, *args):
            planned_inputs.append(args)

    class FakeRoutePlanner:
        def __init__(self, planner):
            self.planner = planner

        def plan(self, route, timeout):
            route_calls.append((route, timeout))
            if len(route_calls) == 1:
                raise RuntimeError("staging waypoint is invalid")
            return sentinel_plan

    monkeypatch.setattr(
        simulation_module.MeshExtractor,
        "blanket_meshes",
        staticmethod(lambda _: blanket_meshes),
    )
    monkeypatch.setattr(
        simulation_module.MeshExtractor,
        "vacuum_vessel_meshes",
        staticmethod(lambda _: vacuum_vessel_meshes),
    )
    monkeypatch.setattr(
        simulation_module.EnvironmentBuilder,
        "create_sector_walls",
        staticmethod(lambda _: sector_walls),
    )
    monkeypatch.setattr(
        simulation_module.EnvironmentBuilder,
        "create_environment",
        staticmethod(lambda vessels, walls: environment),
    )
    monkeypatch.setattr(simulation_module, "PathAnimator", FakeAnimator)
    monkeypatch.setattr(
        simulation_module.PayloadBuilder,
        "create_payload",
        staticmethod(lambda mesh: payloads.append(mesh.name) or FakeCollisionObject()),
    )
    monkeypatch.setattr(simulation_module, "SinglePathPlanner", FakeSinglePathPlanner)
    monkeypatch.setattr(simulation_module, "RoutePlanner", FakeRoutePlanner)

    simulation = BlanketRemovalSimulation(None, None, n_TF=16)
    simulation.run_isolated_removal(timeout=12)

    np.testing.assert_equal(
        [route.start.name for route, _ in route_calls],
        ["starting position", "starting position"],
    )
    np.testing.assert_equal(
        [route.waypoints[1].name for route, _ in route_calls],
        ["below_upper_port", "below_upper_port"],
    )
    np.testing.assert_equal([timeout for _, timeout in route_calls], [12, 12])
    np.testing.assert_equal(payloads, ["OBS_FW_0 1", "IBS_FW_0 1"])
    np.testing.assert_equal(
        [call[1] for call in animator_calls["waypoints"]],
        [["vacuum_vessel"], ["vacuum_vessel"]],
    )
    np.testing.assert_equal(
        animator_calls["animations"],
        [("IBS_FW_0 1", ["vacuum_vessel"], sentinel_plan)],
    )
    np.testing.assert_equal(
        [args[3] for args in planned_inputs],
        [environment, environment],
    )
    np.testing.assert_equal(
        "No route found OBS_FW_0 1: staging waypoint is invalid"
        in caplog.text,
        True,
    )
