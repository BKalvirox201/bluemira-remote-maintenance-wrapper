from __future__ import annotations

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from bluemira.geometry import BluemiraGeo


@dataclass
class DecompositionResultList:
    decomposition_results: list[DecompositionResult]

    def translate(self, vector: tuple[float, float, float]) -> None:
        for result in self.decomposition_results:
            result.translate(vector)

    def scale(self, factor: float) -> None:
        for result in self.decomposition_results:
            result.scale(factor)

    def rotate(
        self,
        base: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        degree: float = 180.0,
    ) -> None:
        for result in self.decomposition_results:
            result.rotate(base, direction, degree)

    # TODO: Come up with a good set of copy functions

    @staticmethod
    def plot_all_results(
        results: DecompositionResultList, transparency: float = 1
    ) -> None:
        figure = plt.figure(figsize=(14, 14))
        axes = figure.add_subplot(111, projection="3d")
        colours = plt.cm.tab20.colors
        colour_index = 0
        all_points = []
        for result in results.decomposition_results:
            for hull in result.sub_hulls:
                colour = colours[colour_index % len(colours)]
                colour_index += 1
                points = hull.points
                all_points.extend(points)
                for simplex in hull.simplices:
                    triangle = points[simplex]
                    axes.add_collection3d(
                        Poly3DCollection(
                            [triangle],
                            facecolor=colour,
                            edgecolor="black",
                            alpha=transparency,
                        )
                    )

        all_points = np.asarray(all_points)
        axes.set_box_aspect(
            (
                max(np.ptp(all_points[:, 0]), 1e-6),
                max(np.ptp(all_points[:, 1]), 1e-6),
                max(np.ptp(all_points[:, 2]), 1e-6),
            )
        )
        plt.tight_layout()
        plt.show()


@dataclass
class DecompositionResult:
    component_name: str
    component_geo: BluemiraGeo
    sub_hulls: list[ConvexHull]

    def _transform_hulls(self, transform_function) -> None:
        self.sub_hulls = [
            ConvexHull(transform_function(hull.points)) for hull in self.sub_hulls
        ]

    def translate(self, vector: tuple[float, float, float]) -> None:
        translation_vector = np.asarray(vector, dtype=float)
        self.component_geo.translate(vector)
        self._transform_hulls(lambda points: points + translation_vector)

    def scale(self, factor: float) -> None:
        self.component_geo.scale(factor)
        self._transform_hulls(lambda points: points * factor)

    def rotate(
        self,
        base: tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        degree: float = 180.0,
    ) -> None:
        self.component_geo.rotate(base, direction, degree)
        rotation_matrix = self._rotation_matrix(direction, degree)
        rotation_origin = np.asarray(base, dtype=float)
        self._transform_hulls(
            lambda points: ((points - rotation_origin) @ rotation_matrix.T)
            + rotation_origin
        )

    @staticmethod
    def _rotation_matrix(
        direction: tuple[float, float, float], degree: float
    ) -> np.ndarray:
        axis = np.asarray(direction, dtype=float)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(degree)
        x, y, z = axis
        cosine = np.cos(angle)
        sine = np.sin(angle)
        one_minus_cosine = 1.0 - cosine
        return np.array(
            [
                [
                    cosine + x**2 * one_minus_cosine,
                    x * y * one_minus_cosine - z * sine,
                    x * z * one_minus_cosine + y * sine,
                ],
                [
                    y * x * one_minus_cosine + z * sine,
                    cosine + y**2 * one_minus_cosine,
                    y * z * one_minus_cosine - x * sine,
                ],
                [
                    z * x * one_minus_cosine - y * sine,
                    z * y * one_minus_cosine + x * sine,
                    cosine + z**2 * one_minus_cosine,
                ],
            ]
        )

    def copy(self, component_name: str | None = None) -> "DecompositionResult":
        return DecompositionResult(
            component_name=component_name or self.component_name,
            component_geo=self.component_geo.copy(),
            sub_hulls=[ConvexHull(hull.points.copy()) for hull in self.sub_hulls],
        )

    def deepcopy(self, component_name: str | None = None) -> "DecompositionResult":
        return DecompositionResult(
            component_name=component_name or self.component_name,
            component_geo=self.component_geo.deepcopy(),
            sub_hulls=[
                ConvexHull(copy.deepcopy(hull.points)) for hull in self.sub_hulls
            ],
        )

    def plot_convex_hulls(self) -> None:
        figure = plt.figure(figsize=(12, 12))
        axes = figure.add_subplot(111, projection="3d")
        colours = plt.cm.tab20.colors
        all_points = []
        for hull_index, hull in enumerate(self.sub_hulls):
            colour = colours[hull_index % len(colours)]
            points = hull.points
            all_points.extend(points)
            for simplex in hull.simplices:
                triangle = points[simplex]
                axes.add_collection3d(
                    Poly3DCollection(
                        [triangle], facecolor=colour, edgecolor="black", alpha=0.5
                    )
                )

        all_points = np.asarray(all_points)
        axes.set_box_aspect(
            (
                max(np.ptp(all_points[:, 0]), 1e-6),
                max(np.ptp(all_points[:, 1]), 1e-6),
                max(np.ptp(all_points[:, 2]), 1e-6),
            )
        )

        axes.set_title(self.component_name)
        plt.tight_layout()
        plt.show()


# TODO:
# Yeah the convexhulldecomposition is cool in 2d, and the blanket decomposer does this.. but also
# Basically need to write a 3d voxel decomposer.. or use something that already does it.


class ConvexHullDecomposition:
    def run(
        self,
        profile: np.ndarray,
        show_plots: bool = False,
        title: str = "Convex Decomposition",
    ) -> list[np.ndarray]:
        triangles = self.triangulate_ear_clipping(profile)
        adjacency = self.build_triangle_adjacency(triangles)
        polygons: list[np.ndarray] = []
        consumed: set[int] = set()

        for triangle_index, neighbours in adjacency.items():
            if triangle_index in consumed:
                continue

            merged = False
            for neighbour_index in neighbours:
                if neighbour_index in consumed:
                    continue

                candidate = self.merge_triangles(
                    triangles[triangle_index], triangles[neighbour_index]
                )

                if not self.is_convex_polygon(candidate):
                    continue

                polygons.append(candidate)
                consumed.add(triangle_index)
                consumed.add(neighbour_index)
                merged = True
                break

            if not merged:
                polygons.append(triangles[triangle_index])

        if show_plots:
            self.plot_polygons(profile=profile, polygons=polygons, title=title)

        return polygons

    @staticmethod
    def polygon_area_signed(polygon: np.ndarray) -> float:
        x_coordinates = polygon[:, 0]
        y_coordinates = polygon[:, 1]

        return 0.5 * (
            np.dot(x_coordinates, np.roll(y_coordinates, -1))
            - np.dot(y_coordinates, np.roll(x_coordinates, -1))
        )

    @staticmethod
    def cross_product_z(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> float:
        return (point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (
            point_b[1] - point_a[1]
        ) * (point_c[0] - point_a[0])

    @staticmethod
    def point_in_triangle(
        point: np.ndarray,
        triangle: np.ndarray,
    ) -> bool:
        point_a, point_b, point_c = triangle

        vector_0 = point_c - point_a
        vector_1 = point_b - point_a
        vector_2 = point - point_a

        dot_00 = np.dot(
            vector_0,
            vector_0,
        )

        dot_01 = np.dot(
            vector_0,
            vector_1,
        )

        dot_02 = np.dot(
            vector_0,
            vector_2,
        )

        dot_11 = np.dot(
            vector_1,
            vector_1,
        )

        dot_12 = np.dot(
            vector_1,
            vector_2,
        )

        denominator = dot_00 * dot_11 - dot_01 * dot_01

        if np.isclose(
            denominator,
            0.0,
        ):
            return False

        u_value = (dot_11 * dot_02 - dot_01 * dot_12) / denominator

        v_value = (dot_00 * dot_12 - dot_01 * dot_02) / denominator

        return u_value >= 0.0 and v_value >= 0.0 and (u_value + v_value) <= 1.0

    def triangulate_ear_clipping(
        self,
        polygon: np.ndarray,
    ) -> list[np.ndarray]:
        vertices = np.asarray(
            polygon,
            dtype=float,
        )

        area = self.polygon_area_signed(vertices)

        if area < 0.0:
            vertices = vertices[::-1]

        vertex_indices = list(range(len(vertices)))

        triangles: list[np.ndarray] = []

        while len(vertex_indices) > 3:
            ear_found = False

            for (
                position,
                current_index,
            ) in enumerate(vertex_indices):
                previous_index = vertex_indices[position - 1]

                next_index = vertex_indices[(position + 1) % len(vertex_indices)]

                point_a = vertices[previous_index]

                point_b = vertices[current_index]

                point_c = vertices[next_index]

                if (
                    self.cross_product_z(
                        point_a,
                        point_b,
                        point_c,
                    )
                    <= 0.0
                ):
                    continue

                candidate_triangle = np.array(
                    [
                        point_a,
                        point_b,
                        point_c,
                    ]
                )

                contains_point = False

                for test_index in vertex_indices:
                    if test_index in (
                        previous_index,
                        current_index,
                        next_index,
                    ):
                        continue

                    if self.point_in_triangle(
                        vertices[test_index],
                        candidate_triangle,
                    ):
                        contains_point = True
                        break

                if contains_point:
                    continue

                triangles.append(candidate_triangle)

                vertex_indices.remove(current_index)

                ear_found = True

                break

            if not ear_found:
                raise RuntimeError("Ear clipping failed.")

        triangles.append(vertices[vertex_indices])

        return triangles

    @staticmethod
    def canonical_point(
        point: np.ndarray,
        decimals: int = 8,
    ) -> tuple[float, float]:
        return (
            round(
                float(point[0]),
                decimals,
            ),
            round(
                float(point[1]),
                decimals,
            ),
        )

    def get_triangle_edges(
        self,
        triangle: np.ndarray,
    ) -> set:
        edges = set()

        for index in range(3):
            point_a = self.canonical_point(triangle[index])

            point_b = self.canonical_point(triangle[(index + 1) % 3])

            edges.add(
                frozenset(
                    [
                        point_a,
                        point_b,
                    ]
                )
            )

        return edges

    def build_triangle_adjacency(
        self,
        triangles: list[np.ndarray],
    ) -> dict[int, list[int]]:
        adjacency = {index: [] for index in range(len(triangles))}

        for index_a in range(len(triangles)):
            edges_a = self.get_triangle_edges(triangles[index_a])

            for index_b in range(
                index_a + 1,
                len(triangles),
            ):
                edges_b = self.get_triangle_edges(triangles[index_b])

                if len(edges_a & edges_b) != 1:
                    continue

                adjacency[index_a].append(index_b)

                adjacency[index_b].append(index_a)

        return adjacency

    @staticmethod
    def merge_triangles(
        triangle_a: np.ndarray,
        triangle_b: np.ndarray,
    ) -> np.ndarray:
        vertices = np.vstack(
            [
                triangle_a,
                triangle_b,
            ]
        )

        unique_vertices = []

        for vertex in vertices:
            if not any(
                np.allclose(
                    vertex,
                    existing,
                )
                for existing in unique_vertices
            ):
                unique_vertices.append(vertex)

        unique_vertices = np.asarray(unique_vertices)

        centre = unique_vertices.mean(axis=0)

        angles = np.arctan2(
            unique_vertices[:, 1] - centre[1],
            unique_vertices[:, 0] - centre[0],
        )

        ordered_indices = np.argsort(angles)

        return unique_vertices[ordered_indices]

    @staticmethod
    def is_convex_polygon(
        polygon: np.ndarray,
        tolerance: float = 1e-10,
    ) -> bool:
        vertex_count = len(polygon)

        if vertex_count < 4:
            return True

        orientation_sign = None

        for index in range(vertex_count):
            point_a = polygon[index]

            point_b = polygon[(index + 1) % vertex_count]

            point_c = polygon[(index + 2) % vertex_count]

            cross_product = (point_b[0] - point_a[0]) * (point_c[1] - point_b[1]) - (
                point_b[1] - point_a[1]
            ) * (point_c[0] - point_b[0])

            if abs(cross_product) < tolerance:
                continue

            current_sign = cross_product > 0.0

            if orientation_sign is None:
                orientation_sign = current_sign

                continue

            if current_sign != orientation_sign:
                return False

        return True

    def plot_polygons(
        self, profile: np.ndarray, polygons: list[np.ndarray], title: str
    ) -> None:
        figure, axes = plt.subplots(figsize=(10, 10))
        colours = plt.cm.tab20.colors

        for index, polygon in enumerate(polygons):
            closed_polygon = np.vstack([polygon, polygon[0]])

            axes.fill(
                closed_polygon[:, 0],
                closed_polygon[:, 1],
                color=colours[index % len(colours)],
                alpha=0.6,
            )
            axes.plot(closed_polygon[:, 0], closed_polygon[:, 1], color="black")

        closed_profile = np.vstack([profile, profile[0]])
        axes.plot(
            closed_profile[:, 0], closed_profile[:, 1], color="black", linewidth=3
        )

        axes.set_aspect("equal")
        axes.set_title(title)

        plt.tight_layout()
        plt.show()
