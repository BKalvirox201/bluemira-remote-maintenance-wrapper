from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Iterable, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, KDTree
from tqdm import tqdm

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.reactor import Reactor
from bluemira.geometry.base import BluemiraGeo, BoundingBox
from bluemira_wrapper.geometry_overlap_detection.lib.gjk_collision_detection import (
    GJKCollisionDetection,
)

if TYPE_CHECKING:
    from Part import Shape  # type: ignore

IndexPairArray: TypeAlias = NDArray[np.int32]
FloatArray: TypeAlias = NDArray[np.float64]
NamedCollisionPair: TypeAlias = tuple[str, str]


class GeometryData(NamedTuple):
    """Names, shapes."""

    names: list[str]
    geometrys: list[BluemiraGeo]


class GeometryExtractor:
    """Flatten reactor geometry."""

    @staticmethod
    def extract(reactor: Reactor) -> GeometryData:
        params = reactor._init_construction_param_values(None, {})
        tree = reactor._build_component_tree("xyz", params)

        names: list[str] = []
        geometries: list[BluemiraGeo] = []

        physical_components = GeometryExtractor.extract_physical_components([tree])
        for index, component in enumerate(physical_components):
            names.append(f"{component.name}_{index}")
            geometries.append(component.shape)

        return GeometryData(names, geometries)

    @staticmethod
    def extract_from_components(components: list[Component]) -> GeometryData:
        names: list[str] = []
        geometries: list[BluemiraGeo] = []

        physical_components = GeometryExtractor.extract_physical_components(components)
        for index, component in enumerate(physical_components):
            names.append(f"{component.name}_{index}")
            geometries.append(component.shape)

        return GeometryData(names, geometries)

    @staticmethod
    def extract_physical_components(
        components: list[Component],
    ) -> list[PhysicalComponent]:
        def collect(root: Component) -> list[Component]:
            output: list[Component] = []
            stack: list[Component] = [root]

            while stack:
                node = stack.pop()
                output.append(node)
                stack.extend(node.children)

            return output

        return [
            component
            for root in components
            for component in collect(root)
            if isinstance(component, PhysicalComponent)
        ]


class OverlapUtilities:
    @staticmethod
    def compute_bounding_box_centres(boxes: list[BoundingBox]) -> FloatArray:
        return np.asarray(
            [
                [
                    (box.x_min + box.x_max) * 0.5,
                    (box.y_min + box.y_max) * 0.5,
                    (box.z_min + box.z_max) * 0.5,
                ]
                for box in boxes
            ],
            dtype=float,
        )

    @staticmethod
    def collate_bounding_box_bounds(
        boxes: list[BoundingBox],
    ) -> tuple[FloatArray, FloatArray]:
        minimums = np.asarray(
            [[box.x_min, box.y_min, box.z_min] for box in boxes], dtype=float
        )
        maximums = np.asarray(
            [[box.x_max, box.y_max, box.z_max] for box in boxes], dtype=float
        )

        return minimums, maximums

    @staticmethod
    def remove_non_overlapping_pairs(
        pairs: IndexPairArray,
        minimums: FloatArray,
        maximums: FloatArray,
    ) -> IndexPairArray:
        if pairs.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        overlapping_pairs: list[tuple[int, int]] = []
        for first, second in pairs:
            overlap = np.minimum(maximums[first], maximums[second]) - np.maximum(
                minimums[first], minimums[second]
            )
            overlap = np.maximum(overlap, 0.0)
            if np.all(overlap > 0):
                overlapping_pairs.append((int(first), int(second)))

        if not overlapping_pairs:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(overlapping_pairs, dtype=np.int32)

    @staticmethod
    def determine_cad_overlaps(
        pairs: IndexPairArray,
        geometrys: list[BluemiraGeo],
        count_touching_as_collision: bool,
    ) -> IndexPairArray:
        overlaps: list[tuple[int, int]] = []

        for first, second in tqdm(
            pairs,
            total=len(pairs),
            desc="Checking overlaps",
        ):
            common = geometrys[first].shape.common(geometrys[second].shape)
            if common.isNull():
                continue

            volume = float(getattr(common, "Volume", 0.0))
            if count_touching_as_collision:
                overlaps.append((int(first), int(second)))
            elif volume > 0.0:
                overlaps.append((int(first), int(second)))

        if not overlaps:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(overlaps, dtype=np.int32)


class GJKOverlapDetector:
    @classmethod
    def detect(
        cls,
        geometry: GeometryData,
        count_touching_as_collision: bool = False,
    ) -> list[NamedCollisionPair]:
        print("Starting GjK Overlap Detection")

        boxes = [geometry_shape.bounding_box for geometry_shape in geometry.geometrys]
        pairs = KDTreePairFinder.find_pairs(boxes)
        minimums, maximums = OverlapUtilities.collate_bounding_box_bounds(boxes)
        pairs = OverlapUtilities.remove_non_overlapping_pairs(pairs, minimums, maximums)

        hull_vertices = [
            geometry_shape.vertexes.xyz.T
            for geometry_shape in tqdm(
                geometry.geometrys,
                desc="Building convex hulls",
            )
        ]

        overlaps: list[NamedCollisionPair] = []
        for first_index, second_index in tqdm(
            pairs,
            total=len(pairs),
            desc="Running GJK",
        ):
            collision = GJKCollisionDetection.gjk_collision(
                hull_vertices[first_index],
                hull_vertices[second_index],
                count_touching_as_collision=count_touching_as_collision,
            )

            if collision:
                overlaps.append(
                    (geometry.names[first_index], geometry.names[second_index])
                )

        return overlaps


class KDTreePairFinder:
    @staticmethod
    def compute_bounding_sphere_radius(
        boxes: list[BoundingBox],
    ) -> FloatArray:
        extents = np.asarray(
            [
                [
                    box.x_max - box.x_min,
                    box.y_max - box.y_min,
                    box.z_max - box.z_min,
                ]
                for box in boxes
            ],
            dtype=float,
        )

        return 0.5 * np.linalg.norm(extents, axis=1)

    @classmethod
    def find_pairs(cls, boxes: list[BoundingBox]) -> IndexPairArray:
        centres = OverlapUtilities.compute_bounding_box_centres(boxes)
        if centres.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)

        radii = cls.compute_bounding_sphere_radius(boxes)
        maximum_radius = float(np.max(radii))

        tree = KDTree(centres)
        candidate_pairs: list[tuple[int, int]] = []

        for source_index in range(len(centres)):
            neighbours = tree.query_ball_point(
                centres[source_index], radii[source_index] + maximum_radius
            )

            candidate_pairs.extend(
                (source_index, target_index)
                for target_index in neighbours
                if source_index < target_index
            )

        if not candidate_pairs:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(candidate_pairs, dtype=np.int32)


class KDTreeOverlapDetector:
    @classmethod
    def detect(
        cls, geometry: GeometryData, count_touching_as_collision: bool = False
    ) -> list[NamedCollisionPair]:
        print("Starting KD Tree Overlap Detection")

        boxes = [geometry_shape.bounding_box for geometry_shape in geometry.geometrys]
        pairs = KDTreePairFinder.find_pairs(boxes)
        minimums, maximums = OverlapUtilities.collate_bounding_box_bounds(boxes)

        pairs = OverlapUtilities.remove_non_overlapping_pairs(pairs, minimums, maximums)
        overlaps = OverlapUtilities.determine_cad_overlaps(
            pairs, geometry.geometrys, count_touching_as_collision
        )

        return [
            (geometry.names[first_index], geometry.names[second_index])
            for first_index, second_index in overlaps
        ]


class SpatialGridOverlapDetector:
    @staticmethod
    def build_grid(
        centres: FloatArray, cell_size: float
    ) -> dict[tuple[int, int, int], list[int]]:
        grid: dict[tuple[int, int, int], list[int]] = {}

        keys = np.floor(centres / cell_size).astype(int)
        for index, key in enumerate(keys):
            grid.setdefault((key[0], key[1], key[2]), []).append(index)

        return grid

    @staticmethod
    def grid_pairs(
        grid: dict[tuple[int, int, int], list[int]],
    ) -> IndexPairArray:
        pairs: set[tuple[int, int]] = set()

        for (x_coord, y_coord, z_coord), indices in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbour = (x_coord + dx, y_coord + dy, z_coord + dz)

                        if neighbour not in grid:
                            continue

                        for first_index in indices:
                            for second_index in grid[neighbour]:
                                if first_index < second_index:
                                    pairs.add((first_index, second_index))

        if not pairs:
            return np.empty((0, 2), dtype=np.int32)

        return np.asarray(list(pairs), dtype=np.int32)

    @classmethod
    def detect(
        cls, geometry: GeometryData, count_touching_as_collision: bool = False
    ) -> list[NamedCollisionPair]:
        print("Starting Spatial Grid Overlap Detection")

        boxes = [geometry_shape.bounding_box for geometry_shape in geometry.geometrys]
        centres = OverlapUtilities.compute_bounding_box_centres(boxes)
        minimums, maximums = OverlapUtilities.collate_bounding_box_bounds(boxes)

        sizes = np.asarray([box.x_max - box.x_min for box in boxes], dtype=float)
        cell_size = float(np.mean(sizes) * 2.0)

        grid = cls.build_grid(centres, cell_size)
        pairs = cls.grid_pairs(grid)
        pairs = OverlapUtilities.remove_non_overlapping_pairs(pairs, minimums, maximums)
        overlaps = OverlapUtilities.determine_cad_overlaps(
            pairs, geometry.geometrys, count_touching_as_collision
        )

        return [
            (geometry.names[first_index], geometry.names[second_index])
            for first_index, second_index in overlaps
        ]


def fprint_overlaps(
    overlaps: Iterable[NamedCollisionPair],
) -> None:
    counter: Counter[NamedCollisionPair] = Counter()
    pattern = re.compile(r"[\s_]*\d+(?:_\d+)*$")

    for first_name, second_name in overlaps:
        left_name = re.sub(pattern, "", first_name).strip()
        right_name = re.sub(pattern, "", second_name).strip()

        key: NamedCollisionPair = (
            (left_name, right_name)
            if left_name <= right_name
            else (right_name, left_name)
        )
        counter[key] += 1

    print(f"Total overlaps = {sum(counter.values())}")
    for (left_name, right_name), count in counter.items():
        print(f"{left_name}, {right_name}: {count}")
