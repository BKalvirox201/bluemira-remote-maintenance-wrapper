from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from .three_d_shape_decomposition.convex_hull_decomposition import (
    DecompositionResultList,
)


class GJKCollisionDetection:
    @classmethod
    def _support(
        cls, vertices_a: np.ndarray, vertices_b: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        def _furthest_point(vertices: np.ndarray, direction: np.ndarray) -> np.ndarray:
            index = np.argmax(vertices @ direction)
            return vertices[index]

        point_a = _furthest_point(vertices_a, direction)
        point_b = _furthest_point(vertices_b, -direction)
        return point_a - point_b

    @classmethod
    def _process_line(cls, simplex: list[np.ndarray]) -> tuple[bool, np.ndarray]:
        point_a = simplex[-1]
        point_b = simplex[-2]

        vector_ab = point_b - point_a
        vector_ao = -point_a

        if np.dot(vector_ab, vector_ao) > 0:
            direction = np.cross(np.cross(vector_ab, vector_ao), vector_ab)
        else:
            simplex[:] = [point_a]
            direction = vector_ao

        return False, direction

    @classmethod
    def _process_triangle(cls, simplex: list[np.ndarray]) -> tuple[bool, np.ndarray]:
        point_a = simplex[-1]
        point_b = simplex[-2]
        point_c = simplex[-3]

        vector_ab = point_b - point_a
        vector_ac = point_c - point_a
        vector_ao = -point_a

        normal_abc = np.cross(vector_ab, vector_ac)

        ac_perp = np.cross(normal_abc, vector_ac)
        if np.dot(ac_perp, vector_ao) > 0:
            simplex[:] = [point_c, point_a]
            direction = np.cross(np.cross(vector_ac, vector_ao), vector_ac)
            return False, direction

        ab_perp = np.cross(vector_ab, normal_abc)
        if np.dot(ab_perp, vector_ao) > 0:
            simplex[:] = [point_b, point_a]
            return cls._process_line(simplex)

        if np.dot(normal_abc, vector_ao) > 0:
            return False, normal_abc

        simplex[:] = [point_b, point_c, point_a]
        return False, -normal_abc

    @classmethod
    def _process_tetrahedron(
        cls, simplex: list[np.ndarray], tolerance: float = 1e-9
    ) -> tuple[bool, np.ndarray, bool]:
        point_a = simplex[-1]
        point_b = simplex[-2]
        point_c = simplex[-3]
        point_d = simplex[-4]

        vector_ao = -point_a

        normal_abc = np.cross(point_b - point_a, point_c - point_a)
        normal_acd = np.cross(point_c - point_a, point_d - point_a)
        normal_adb = np.cross(point_d - point_a, point_b - point_a)

        # TODO: Implement a proper EPA
        touching = (
            abs(np.dot(normal_abc, vector_ao)) <= tolerance
            or abs(np.dot(normal_acd, vector_ao)) <= tolerance
            or abs(np.dot(normal_adb, vector_ao)) <= tolerance
        )

        if np.dot(normal_abc, vector_ao) > tolerance:
            simplex[:] = [point_c, point_b, point_a]
            return False, normal_abc, touching

        if np.dot(normal_acd, vector_ao) > tolerance:
            simplex[:] = [point_d, point_c, point_a]
            return False, normal_acd, touching

        if np.dot(normal_adb, vector_ao) > tolerance:
            simplex[:] = [point_b, point_d, point_a]
            return False, normal_adb, touching

        return True, np.zeros(3), touching

    @classmethod
    def _update_simplex(
        cls, simplex: list[np.ndarray]
    ) -> tuple[bool, np.ndarray, bool]:
        if len(simplex) == 2:
            contains_origin, direction = cls._process_line(simplex)
            return contains_origin, direction, False

        if len(simplex) == 3:
            # TODO: It is feasable for two triangles to touch, fix
            contains_origin, direction = cls._process_triangle(simplex)
            return contains_origin, direction, False

        return cls._process_tetrahedron(simplex)

    @classmethod
    def gjk_collision(
        cls,
        vertices_a: np.ndarray,
        vertices_b: np.ndarray,
        count_touching_as_collision: bool = True,
        tolerance: float = 1e-9,
        max_iterations: int = 50,
    ) -> bool:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
        simplex = [cls._support(vertices_a, vertices_b, direction)]
        direction = -simplex[0]

        for _ in range(max_iterations):
            point = cls._support(vertices_a, vertices_b, direction)
            projection = np.dot(point, direction)

            if count_touching_as_collision:
                if projection < -tolerance:
                    return False
            else:
                if projection <= tolerance:
                    return False

            simplex.append(point)
            contains_origin, direction, touching = cls._update_simplex(simplex)

            if contains_origin:
                if touching and not count_touching_as_collision:
                    return False

                return True

            assert direction is not None

        raise RuntimeError(f"GJK failed converge after {max_iterations} iterations")


@dataclass
class CollisionResult:
    shape_a_name: str
    shape_a_index: int
    shape_b_name: str
    shape_b_index: int


def create_bounding_sphere(
    vertices: NDArray[np.number],
) -> tuple[NDArray[np.float64], float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2:
        raise ValueError(f"Expected shape (n, 3), got {vertices.shape}")

    centre = vertices.mean(axis=0)
    squared_distances = np.sum((vertices - centre) ** 2, axis=1)
    radius = float(np.sqrt(np.max(squared_distances)))
    return centre, radius


def spheres_overlap(
    centre_a: NDArray[np.float64],
    radius_a: float,
    centre_b: NDArray[np.float64],
    radius_b: float,
) -> bool:
    radius_sum = radius_a + radius_b
    return bool(
        np.dot(centre_a - centre_b, centre_a - centre_b) <= radius_sum * radius_sum
    )


def find_collisions_between_decomposition_results(
    result_a: DecompositionResultList,
    result_b: DecompositionResultList,
    count_touching_as_collision: bool = True,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
):
    collisions: list[CollisionResult] = []

    # for each shape
    for a_index, a in enumerate(result_a.decomposition_results):
        for b_index, b in enumerate(result_b.decomposition_results):
            # NOTE: Shape prefilter?

            # for each hull in shape
            sub_hulls_collide = False
            for a_subhull in a.sub_hulls:
                if sub_hulls_collide == True:
                    break
                for b_subhull in b.sub_hulls:
                    if sub_hulls_collide == True:
                        break

                    # bounding sphere prefilter
                    centre_a, radius_a = create_bounding_sphere(a_subhull.points)
                    centre_b, radius_b = create_bounding_sphere(b_subhull.points)
                    if not spheres_overlap(centre_a, radius_a, centre_b, radius_b):
                        continue

                    if GJKCollisionDetection.gjk_collision(
                        a_subhull.points,
                        b_subhull.points,
                        count_touching_as_collision,
                        tolerance,
                        max_iterations,
                    ):
                        collisions.append(
                            CollisionResult(
                                a.component_name, a_index, b.component_name, b_index
                            )
                        )
                        sub_hulls_collide = True

    return collisions


if __name__ == "_main_":
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull

    points_a = np.random.randn(100, 3)
    points_b = np.random.randn(100, 3)
    points_b[:, 0] += 0.5

    hull_a = ConvexHull(points_a)
    hull_b = ConvexHull(points_b)

    vertices_a = points_a[hull_a.vertices]
    vertices_b = points_b[hull_b.vertices]

    collision = GJKCollisionDetection.gjk_collision(vertices_a, vertices_b)
    print(f"Collision: {collision}")

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    def plot_convex_hull(axis, points, hull: ConvexHull, colour):
        faces = [points[simplex] for simplex in hull.simplices]

        mesh = Poly3DCollection(
            faces, alpha=0.3, facecolor=colour, edgecolor="k", linewidth=0.5
        )

        axis.add_collection3d(mesh)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=colour, s=10)

    plot_convex_hull(axis, points_a, hull_a, "tab:blue")
    plot_convex_hull(axis, points_b, hull_b, "tab:red")

    all_points = np.vstack([points_a, points_b])

    axis.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
    axis.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
    axis.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")

    axis.set_title(f"GJK Collision: {collision}")

    plt.tight_layout()
    plt.show()
