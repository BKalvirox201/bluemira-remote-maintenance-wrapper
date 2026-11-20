from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation, Slerp

from .models import Mesh, PlannedPath, PlannedRoute, WaypointRoute


class PathAnimator:
    @staticmethod
    def add_world_axes(plotter: pv.Plotter, scale: float = 5.0) -> None:
        origin = np.zeros(3)
        plotter.add_mesh(pv.Line(origin, [scale, 0, 0]), color="red", line_width=3)
        plotter.add_mesh(pv.Line(origin, [0, scale, 0]), color="green", line_width=3)
        plotter.add_mesh(pv.Line(origin, [0, 0, scale]), color="blue", line_width=3)

    @staticmethod
    def axis_lines(
        origin: np.ndarray, rotation: np.ndarray, scale: float = 2.0
    ) -> list[pv.PolyData]:
        return [
            pv.Line(origin, origin + rotation[:, axis] * scale) for axis in range(3)
        ]

    @staticmethod
    def mesh_to_pyvista(mesh: Mesh) -> pv.PolyData:
        if len(mesh.triangles) == 0:
            raise ValueError(f"Mesh '{mesh.name}' contains no triangles.")
        faces = np.column_stack(
            (
                np.full(len(mesh.triangles), 3),
                mesh.triangles,
            )
        ).ravel()
        return pv.PolyData(mesh.vertices, faces)

    @staticmethod
    def combine_meshes(meshes: list[Mesh]) -> Mesh:
        vertex_blocks = []
        triangle_blocks = []
        vertex_offset = 0

        for mesh in meshes:
            vertex_blocks.append(mesh.vertices)
            triangle_blocks.append(mesh.triangles + vertex_offset)
            vertex_offset += len(mesh.vertices)

        return Mesh(
            name=",".join(mesh.name for mesh in meshes),
            vertices=np.vstack(vertex_blocks),
            triangles=np.vstack(triangle_blocks),
            pivot_point=np.mean([mesh.pivot_point for mesh in meshes], axis=0),
        )

    @staticmethod
    def state_position(state) -> np.ndarray:
        return np.array([state.getX(), state.getY(), state.getZ()], dtype=float)

    @staticmethod
    def state_rotation(state) -> np.ndarray:
        quaternion = state.rotation()
        return Rotation.from_quat(
            [
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w,
            ]
        ).as_matrix()

    @staticmethod
    def create_pose(
        position: np.ndarray, rotation: np.ndarray, pivot_point: np.ndarray
    ) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position - rotation @ pivot_point
        return transform

    def show_waypoints(
        self, payload_mesh: Mesh, environment_meshes: list[Mesh], route: WaypointRoute
    ):
        plotter = pv.Plotter()
        # self.add_world_axes(plotter)
        plotter.add_text(
            "World axes: X red, Y green, Z blue\nStaging pose: cyan ghost",
            position="upper_left",
            font_size=12,
        )
        for mesh in environment_meshes:
            plotter.add_mesh(
                self.mesh_to_pyvista(mesh), color="lightgrey", opacity=0.15
            )

        payload_pyvista = self.mesh_to_pyvista(payload_mesh)

        waypoint_points = []
        for waypoint in route.waypoints:
            waypoint_points.append(waypoint.position)
            waypoint_mesh = payload_pyvista.copy()
            if waypoint.rotation_matrix is None:
                raise ValueError(f"Waypoint '{waypoint.name}' has no rotation matrix")

            transform = self.create_pose(
                waypoint.position,
                waypoint.rotation_matrix,
                payload_mesh.pivot_point,
            )
            waypoint_mesh.transform(transform, inplace=True)
            plotter.add_mesh(
                waypoint_mesh,
                color="cyan" if waypoint.is_staging else "yellow",
                opacity=0.25 if waypoint.is_staging else 1,
            )
            plotter.add_point_labels([waypoint.position], [waypoint.name], font_size=12)

        plotter.add_mesh(
            pv.lines_from_points(np.asarray(waypoint_points, dtype=float)),
            color="red",
            line_width=4,
        )
        # plotter.show_grid()
        # plotter.add_axes()
        plotter.show(title="Waypoint Route", window_size=[1800, 1000], cpos="xz")

    def interpolate_states(
        self, states: list, samples_per_metre: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(states) < 2:
            return [
                (self.state_position(state), self.state_rotation(state))
                for state in states
            ]

        segment_lengths: list[float] = []
        for index in range(len(states) - 1):
            position_a = self.state_position(states[index])
            position_b = self.state_position(states[index + 1])
            segment_lengths.append(float(np.linalg.norm(position_b - position_a)))

        interpolated_states: list[tuple[np.ndarray, np.ndarray]] = []
        for index, segment_length in enumerate(segment_lengths):
            current_state = states[index]
            next_state = states[index + 1]

            position_a = self.state_position(current_state)
            position_b = self.state_position(next_state)
            rotation_a = Rotation.from_matrix(self.state_rotation(current_state))
            rotation_b = Rotation.from_matrix(self.state_rotation(next_state))

            slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_quat(
                    np.vstack([rotation_a.as_quat(), rotation_b.as_quat()])
                ),
            )

            sample_count = max(1, int(np.ceil(segment_length * samples_per_metre)))
            for fraction in np.linspace(0.0, 1.0, sample_count, endpoint=False):
                position = position_a * (1.0 - fraction) + position_b * fraction
                rotation = slerp([fraction])[0]
                interpolated_states.append((position, rotation.as_matrix()))

        interpolated_states.append(
            (
                self.state_position(states[-1]),
                self.state_rotation(states[-1]),
            )
        )
        return interpolated_states

    def animate(
        self,
        payload_mesh: Mesh,
        environment_meshes: list[Mesh],
        planned_solution: PlannedPath | PlannedRoute,
        animation_time_s: float = 1.0,
    ) -> None:
        plotter = pv.Plotter()
        # self.add_world_axes(plotter)
        plotter.add_text(
            "Camera: drag to rotate, middle drag to pan, wheel to zoom\n"
            "Space to pause/resume, R to restart",
            position="upper_left",
            font_size=12,
        )

        states = list(planned_solution.states())
        # NOTE: The value for animation_frames_per_meter was determined via trial and error
        #       10 is a good value for the scale we are working at
        animation_frames_per_meter = 10
        animation_states = self.interpolate_states(states, animation_frames_per_meter)
        state_count = len(animation_states)

        for mesh in environment_meshes:
            plotter.add_mesh(
                self.mesh_to_pyvista(mesh), color="lightgrey", opacity=0.15
            )

        payload_pyvista = self.mesh_to_pyvista(payload_mesh)
        path_points = [position for position, _ in animation_states]
        plotter.add_mesh(
            pv.lines_from_points(np.asarray(path_points, dtype=float)),
            color="orange",
            line_width=3,
        )

        start_position, start_rotation = animation_states[0]
        start_pose = self.create_pose(
            start_position, start_rotation, payload_mesh.pivot_point
        )

        payload_actor = plotter.add_mesh(payload_pyvista.copy(), color="red")
        payload_actor.user_matrix = start_pose

        initial_world_rotation = getattr(
            planned_solution, "initial_world_rotation", None
        )
        if initial_world_rotation is None:
            initial_world_rotation = np.eye(3)
        # payload_axes = self.axis_lines(
        #     start_position, start_rotation @ initial_world_rotation
        # )
        # axis_colors = ("red", "green", "blue")
        # for axis_line, color in zip(payload_axes, axis_colors):
        #     plotter.add_mesh(axis_line, color=color, line_width=4)

        frame_duration_ms: int = float(
            animation_time_s * 1000 / state_count
        ).__trunc__()

        playback = {"index": 0, "playing": True}

        def update_payload() -> None:
            if playback["index"] >= state_count:
                playback["playing"] = False
                return

            position, rotation = animation_states[playback["index"]]
            transform = self.create_pose(position, rotation, payload_mesh.pivot_point)

            payload_actor.user_matrix = transform
            # for axis_line, axis in zip(
            #     payload_axes,
            #     self.axis_lines(position, rotation @ initial_world_rotation),
            # ):
            #     axis_line.points = axis.points
            playback["index"] += 1
            plotter.render()

        def update(step) -> None:
            if playback["playing"]:
                update_payload()

        def toggle_playback() -> None:
            playback["playing"] = not playback["playing"]

        def restart() -> None:
            playback["index"] = 0
            playback["playing"] = True
            update_payload()

        plotter.add_key_event("space", toggle_playback)
        plotter.add_key_event("r", restart)

        plotter.add_timer_event(
            max_steps=2**31 - 1,
            duration=frame_duration_ms,
            callback=update,
        )

        plotter.show(
            title=payload_mesh.name,
            window_size=[1800, 1000],
            cpos="xz",
        )
