from __future__ import annotations

import fcl
import FreeCAD
import numpy as np
import Part

from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.mesh_extraction import (
    MeshExtractor,
)
from bluemira_wrapper.fcl_ompl_path_planner.blanket_simulation.components.models import (
    Mesh,
)


class EnvironmentBuilder:
    @staticmethod
    def create_sector_walls(
        number_of_vessel_segments: int,
        radial_length: float = 20.0,
        wall_thickness: float = 1.0,
        wall_height: float = 20.0,
    ) -> list[Mesh]:
        segment_angle = np.deg2rad(360.0 / number_of_vessel_segments)

        first_wall = Part.makeBox(
            radial_length,
            wall_thickness,
            wall_height,
            FreeCAD.Vector(0.0, -wall_thickness, -wall_height / 2.0),
        )

        second_wall = Part.makeBox(
            radial_length,
            wall_thickness,
            wall_height,
            FreeCAD.Vector(0.0, 0.0, -wall_height / 2.0),
        )
        second_wall.rotate(
            FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), np.degrees(segment_angle)
        )

        return [
            MeshExtractor.shape_to_mesh(first_wall, "sector_wall_1"),
            MeshExtractor.shape_to_mesh(second_wall, "sector_wall_2"),
        ]

    @staticmethod
    def create_collision_object(meshes: list[Mesh]) -> fcl.CollisionObject:
        geometry = fcl.BVHModel()
        vertex_count = sum(len(mesh.vertices) for mesh in meshes)
        triangle_count = sum(len(mesh.triangles) for mesh in meshes)
        geometry.beginModel(vertex_count, triangle_count)
        for mesh in meshes:
            geometry.addSubModel(mesh.vertices, mesh.triangles)
        geometry.endModel()

        return fcl.CollisionObject(geometry)

    @staticmethod
    def create_environment(
        vacuum_vessel_meshes: list[Mesh], sector_walls: list[Mesh]
    ) -> list[tuple[str, np.ndarray, fcl.CollisionObject]]:
        return [
            (
                mesh.name,
                mesh.pivot_point.copy(),
                EnvironmentBuilder.create_collision_object([mesh]),
            )
            for mesh in [*vacuum_vessel_meshes, *sector_walls]
        ]


class PayloadBuilder:
    @staticmethod
    def create_payload(mesh: Mesh) -> fcl.CollisionObject:
        return EnvironmentBuilder.create_collision_object([mesh])
