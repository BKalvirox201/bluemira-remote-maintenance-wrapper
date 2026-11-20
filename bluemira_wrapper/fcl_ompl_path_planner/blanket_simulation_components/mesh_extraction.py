from __future__ import annotations

import MeshPart
import numpy as np
import Part
from bluemira.base.components import PhysicalComponent
from bluemira.base.look_and_feel import bluemira_print

from eudemo.blanket import Blanket
from eudemo.vacuum_vessel import VacuumVessel

from .models import Mesh


class MeshExtractor:
    @staticmethod
    def shape_to_mesh(
        shape: Part.Shape,
        label: str,
        linear_deflection: float = 0.1,
        angular_deflection: float = 0.01,
    ) -> Mesh:
        freecad_mesh = MeshPart.meshFromShape(
            Shape=shape,
            LinearDeflection=linear_deflection,
            AngularDeflection=angular_deflection,
            Relative=False,
        )

        vertices = []
        vertex_lookup = {}
        triangles = []

        for facet in freecad_mesh.Facets:
            triangle = []
            for point in facet.Points:
                key = tuple(point)
                if key not in vertex_lookup:
                    vertex_lookup[key] = len(vertices)
                    vertices.append(key)
                triangle.append(vertex_lookup[key])
            triangles.append(triangle)

        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int32)
        pivot_point = vertices.mean(axis=0)

        return Mesh(
            name=label,
            vertices=vertices,
            triangles=triangles,
            pivot_point=pivot_point,
        )

    @staticmethod
    def component_to_mesh(component: PhysicalComponent) -> Mesh:
        return MeshExtractor.shape_to_mesh(component.shape.shape, component.name)

    @staticmethod
    def blanket_meshes(blanket: Blanket) -> list[Mesh]:
        bluemira_print("Extracting Blanket meshes for collision detection")
        blanket_component = blanket.component()
        xyz_component = (
            blanket_component
            .get_component("xyz")
            .get_component("Sector 1")
            .get_component("BB 1")
        )
        meshes = []
        for child in xyz_component.children:
            if isinstance(child, PhysicalComponent):
                meshes.append(MeshExtractor.component_to_mesh(child))
        return meshes

    @staticmethod
    def vacuum_vessel_meshes(vacuum_vessel: VacuumVessel) -> list[Mesh]:
        bluemira_print("Extracting Vacuum Vessel meshes for collision detection")
        vacuum_vessel_component = vacuum_vessel.component()
        xyz_component = vacuum_vessel_component.get_component("xyz").get_component(
            "Sector 1"
        )
        meshes = []
        for child in xyz_component.children:
            if not isinstance(child, PhysicalComponent):
                continue
            if "voidspace" in child.name:
                continue
            meshes.append(MeshExtractor.component_to_mesh(child))
        return meshes
