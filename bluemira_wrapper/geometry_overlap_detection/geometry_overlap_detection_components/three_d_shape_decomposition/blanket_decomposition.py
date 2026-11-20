from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from eudemo.blanket import Blanket
from FreeCAD import Part
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from bluemira.base.components import Component
from bluemira.base.reactor import ComponentManager

from .convex_hull_decomposition import (
    ConvexHullDecomposition,
    DecompositionResult,
    DecompositionResultList,
)


class FixBlanketProfiles:
    """
    This class reaches into the freecad api and gets all of the vertex and edge information
    but we have to reconstruct various parts of it.
    """

    def __init__(self, blanket: ComponentManager) -> None:
        self.blanket = blanket

    def run(self, show_plots: bool = False) -> tuple[tuple[str, list[np.ndarray]]]:
        blanket_component: Component = self.blanket.component()

        xyz_component = (
            blanket_component.get_component("xyz")
            .get_component("Sector 1")
            .get_component("BB 1")
        )

        result = []
        for xyz_child in xyz_component.children:
            profile_faces = self.get_candidate_profile_faces(xyz_child.shape.shape)

            boundary_faces = []
            for _, face in profile_faces:
                boundary_points = self.connect_edges(face)
                if len(boundary_points) == 0:
                    continue
                boundary_faces.append(boundary_points)

            if boundary_faces:
                if show_plots:
                    self.plot_boundary_faces(boundary_faces, xyz_child.name)

                result.append((xyz_child.name, boundary_faces))

        return tuple(result)

    @staticmethod
    def get_candidate_profile_faces(shape) -> list[tuple[int, object]]:
        return [
            (face_index, face)
            for face_index, face in enumerate(shape.Faces)
            if len(face.Vertexes) > 4
        ]

    @staticmethod
    def calculate_maximum_deviation(edge) -> float:
        """
        Because Bsplines have the potential to be curved, even though the ones I have seen in construction are all
        straight, handle this by calculating the amount of curvature as a filter
        """
        probe_points = np.asarray(
            [[point.x, point.y, point.z] for point in edge.discretize(25)], dtype=float
        )

        start_point = probe_points[0]
        end_point = probe_points[-1]
        direction = end_point - start_point
        direction_length = np.linalg.norm(direction)

        if direction_length < 1e-12:
            return 0.0

        direction /= direction_length
        projected_lengths = (probe_points - start_point) @ direction
        projected_points = start_point + np.outer(projected_lengths, direction)
        deviations = np.linalg.norm(probe_points - projected_points, axis=1)

        return float(np.max(deviations))

    def discretize_edge(
        self,
        edge: Part.edge,
        # TODO: These defaults are somewhat arbitrarily decided
        points_per_unit_length: float = 3.0,
        minimum_curve_points: int = 6,
        straight_tolerance: float = 1e-3,
    ) -> np.ndarray:
        """
        There appear to be 3 types of edges, straight, bspline (which could be curved, but in practice seem to
        only represent straight edges) and hyperbolic (curved). The curved edges have to be approximated to a
        list of straight edges in order to create representational boxes for collision detection.
        """
        if isinstance(edge.Curve, Part.Line):
            return np.asarray(
                [[point.X, point.Y, point.Z] for point in edge.Vertexes],
                dtype=float,
            )

        point_count: int
        maximum_deviation = self.calculate_maximum_deviation(edge)
        if maximum_deviation < straight_tolerance:
            point_count = 2

        else:
            point_count = max(
                minimum_curve_points, int(edge.Length * points_per_unit_length)
            )

        return np.asarray(
            [[point.x, point.y, point.z] for point in edge.discretize(point_count)],
            dtype=float,
        )

    def connect_edges(self, freecad_face) -> np.ndarray:
        """
        Create the profile by linking the edges together in order
        """
        remaining_edges = [
            self.discretize_edge(edge) for edge in freecad_face.OuterWire.Edges
        ]

        if not remaining_edges:
            return np.empty((0, 3), dtype=float)

        ordered_edges = [remaining_edges.pop(0)]
        # Connect the edges together, pointing down the chain
        while remaining_edges:
            current_end = ordered_edges[-1][-1]

            best_index = None
            best_reverse = False
            best_gap = np.inf
            for edge_index, edge_points in enumerate(remaining_edges):
                start_gap = np.linalg.norm(edge_points[0] - current_end)
                end_gap = np.linalg.norm(edge_points[-1] - current_end)

                if start_gap < best_gap:
                    best_gap = start_gap
                    best_index = edge_index
                    best_reverse = False

                if end_gap < best_gap:
                    best_gap = end_gap
                    best_index = edge_index
                    best_reverse = True

            edge_points = remaining_edges.pop(best_index)
            if best_reverse:
                edge_points = edge_points[::-1]

            ordered_edges.append(edge_points)

        boundary_points = []
        for edge_index, edge_points in enumerate(ordered_edges):
            if edge_index > 0:
                edge_points = edge_points[1:]

            boundary_points.extend(edge_points)
        boundary_points = np.asarray(boundary_points, dtype=float)

        if np.allclose(boundary_points[0], boundary_points[-1]):
            boundary_points = boundary_points[:-1]

        return boundary_points

    @staticmethod
    def plot_boundary_faces(boundary_faces: list[np.ndarray], title: str) -> None:
        figure = plt.figure(figsize=(12, 12))
        axes = figure.add_subplot(111, projection="3d")
        colours = plt.cm.tab10.colors
        all_points = np.vstack(boundary_faces)

        for face_index, boundary_points in enumerate(boundary_faces):
            polygon = Poly3DCollection(
                [boundary_points],
                alpha=0.5,
                facecolor=colours[face_index % len(colours)],
                edgecolor="black",
            )

            axes.add_collection3d(polygon)
            closed_boundary = np.vstack([boundary_points, boundary_points[0]])

            axes.plot(
                closed_boundary[:, 0],
                closed_boundary[:, 1],
                closed_boundary[:, 2],
                color="black",
                linewidth=1.5,
            )

        axes.set_title(title)
        axes.set_box_aspect(
            (
                max(np.ptp(all_points[:, 0]), 1e-6),
                max(np.ptp(all_points[:, 1]), 1e-6),
                max(np.ptp(all_points[:, 2]), 1e-6),
            )
        )

        plt.tight_layout()
        plt.show()


class DecomposeBlanketProfiles:
    def __init__(self, blanket: Blanket) -> None:
        self.blanket = blanket
        self.fix_blanket_profiles = FixBlanketProfiles(blanket=blanket)
        self.convex_hull_decomposition = ConvexHullDecomposition()

    @staticmethod
    def project_boundary_to_plane(
        boundary_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        centroid = boundary_points.mean(axis=0)
        centred_points = boundary_points - centroid
        _, _, right_vectors = np.linalg.svd(centred_points, full_matrices=False)
        local_x_axis = right_vectors[0]
        local_y_axis = right_vectors[1]

        profile = np.column_stack(
            [centred_points @ local_x_axis, centred_points @ local_y_axis]
        )

        return (profile, centroid, local_x_axis, local_y_axis)

    @staticmethod
    def lift_polygon_to_3d(
        polygon: np.ndarray,
        centroid: np.ndarray,
        local_x_axis: np.ndarray,
        local_y_axis: np.ndarray,
    ) -> np.ndarray:
        return (
            centroid
            + np.outer(polygon[:, 0], local_x_axis)
            + np.outer(polygon[:, 1], local_y_axis)
        )

    @staticmethod
    def rotate_profile_start(profile: np.ndarray, start_index: int) -> np.ndarray:
        return np.vstack([profile[start_index:], profile[:start_index]])

    def align_profiles(
        self, profile_a: np.ndarray, profile_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(profile_a) != len(profile_b):
            raise RuntimeError("Profile vertex counts differ.")

        best_index = 0
        best_error = np.inf
        for candidate_index in range(len(profile_b)):
            rotated = self.rotate_profile_start(profile_b, candidate_index)
            error = np.sum(np.linalg.norm(profile_a - rotated, axis=1))
            if error < best_error:
                best_error = error
                best_index = candidate_index

        profile_b = self.rotate_profile_start(profile_b, best_index)
        reverse_profile = profile_b[::-1]
        reverse_error = np.sum(np.linalg.norm(profile_a - reverse_profile, axis=1))
        forward_error = np.sum(np.linalg.norm(profile_a - profile_b, axis=1))
        if reverse_error < forward_error:
            profile_b = reverse_profile

        return (profile_a, profile_b)

    def run(
        self,
        show_profile_plots: bool = False,
        show_decomposition_plots: bool = False,
        show_overall_decomposition_plot: bool = False,
    ) -> DecompositionResultList:
        fixed_profiles = self.fix_blanket_profiles.run(show_profile_plots)

        results = []
        for component_name, boundary_faces in fixed_profiles:
            assert len(boundary_faces) == 2
            face_a = boundary_faces[0]
            face_b = boundary_faces[1]

            profile_a, centroid_a, local_x_axis_a, local_y_axis_a = (
                self.project_boundary_to_plane(face_a)
            )
            profile_b, centroid_b, local_x_axis_b, local_y_axis_b = (
                self.project_boundary_to_plane(face_b)
            )

            if np.dot(local_x_axis_a, local_x_axis_b) < 0:
                local_x_axis_b *= -1
                profile_b[:, 0] *= -1
            if np.dot(local_y_axis_a, local_y_axis_b) < 0:
                local_y_axis_b *= -1
                profile_b[:, 1] *= -1

            profile_a, profile_b = self.align_profiles(profile_a, profile_b)

            convex_hulls = []
            convex_polygons = self.convex_hull_decomposition.run(
                profile=profile_a, show_plots=False
            )
            for polygon in convex_polygons:
                polygon_a_3d = self.lift_polygon_to_3d(
                    polygon,
                    centroid_a,
                    local_x_axis_a,
                    local_y_axis_a,
                )
                polygon_b_3d = self.lift_polygon_to_3d(
                    polygon,
                    centroid_b,
                    local_x_axis_b,
                    local_y_axis_b,
                )
                points = np.vstack([polygon_a_3d, polygon_b_3d])
                convex_hulls.append(ConvexHull(points))

            cad = self.blanket.component().get_component(component_name).shape
            result = DecompositionResult(component_name, cad, convex_hulls)
            results.append(result)
            if show_decomposition_plots and convex_hulls:
                result.plot_convex_hulls()

        result_list = DecompositionResultList(results)
        if show_overall_decomposition_plot:
            DecompositionResultList.plot_all_results(result_list)

        return result_list
