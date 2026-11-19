import numpy as np

from bluemira.geometry.base import BoundingBox
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon
from bluemira_wrapper.geometry_overlap_detection.geometry_overlap_detection import (
    GeometryData,
    KDTreeOverlapDetector,
    KDTreePairFinder,
    OverlapUtilities,
    SpatialGridOverlapDetector,
)


def make_rectangular_face(width: float, height: float) -> BluemiraFace:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    return BluemiraFace(make_polygon(points, closed=True))


def make_box(width: float, depth: float, height: float):
    return extrude_shape(
        make_rectangular_face(width, depth),
        (0.0, 0.0, height),
    )


class TestOverlapUtilities:
    def test_compute_bounding_box_centres(self):
        boxes = [
            BoundingBox(0, 2, 0, 4, 0, 6),
            BoundingBox(2, 4, 2, 6, 2, 8),
        ]

        centres = OverlapUtilities.compute_bounding_box_centres(boxes)

        np.testing.assert_allclose(
            centres,
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [3.0, 4.0, 5.0],
                ]
            ),
        )

    def test_collate_bounding_box_bounds(self):
        boxes = [
            BoundingBox(0, 3, 1, 4, 2, 5),
            BoundingBox(6, 9, 7, 10, 8, 11),
        ]

        minimums, maximums = OverlapUtilities.collate_bounding_box_bounds(boxes)

        np.testing.assert_array_equal(
            minimums,
            np.array(
                [
                    [0, 1, 2],
                    [6, 7, 8],
                ]
            ),
        )

        np.testing.assert_array_equal(
            maximums,
            np.array(
                [
                    [3, 4, 5],
                    [9, 10, 11],
                ]
            ),
        )

    def test_remove_non_overlapping_pairs(self):
        minimums = np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [5, 5, 5],
            ]
        )

        maximums = np.array(
            [
                [1, 1, 1],
                [1.5, 1.5, 1.5],
                [6, 6, 6],
            ]
        )

        pairs = np.array(
            [
                [0, 1],
                [0, 2],
                [1, 2],
            ],
            dtype=np.int32,
        )

        result = OverlapUtilities.remove_non_overlapping_pairs(
            pairs, minimums, maximums, 1e-5
        )

        np.testing.assert_array_equal(result, np.array([[0, 1]], dtype=np.int32))

    def test_remove_non_overlapping_pairs_empty_input(self):
        pairs = np.empty((0, 2), dtype=np.int32)

        result = OverlapUtilities.remove_non_overlapping_pairs(
            pairs, np.empty((0, 3)), np.empty((0, 3)), 1e-5
        )

        assert result.shape == (0, 2)


class TestKDTreePairFinder:
    def test_compute_bounding_sphere_radius(self):
        boxes = [
            BoundingBox(0, 2, 0, 0, 0, 0),
            BoundingBox(0, 2, 0, 2, 0, 2),
        ]

        radii = KDTreePairFinder.compute_bounding_sphere_radius(boxes)
        np.testing.assert_allclose(
            radii,
            [1.0, np.sqrt(12) / 2],
        )

    def test_find_pairs(self):
        boxes = [
            BoundingBox(0, 1, 0, 1, 0, 1),
            BoundingBox(0.5, 1.5, 0, 1, 0, 1),
            BoundingBox(10, 11, 0, 1, 0, 1),
        ]

        pairs = KDTreePairFinder.find_pairs(boxes)
        assert len(pairs) > 0


class TestKDTreeOverlapDetector:
    def test_detect_finds_overlap(self):
        box_1 = make_box(1.0, 1.0, 1.0)
        box_2 = make_box(1.0, 1.0, 1.0)

        box_2.translate((0.5, 0.5, 0.5))

        geometry = GeometryData(
            names=["a", "b"],
            geometrys=[box_1, box_2],
        )

        overlaps = KDTreeOverlapDetector.detect(geometry)
        assert overlaps == [("a", "b")]

    def test_detect_no_overlap(self):
        box_1 = make_box(1.0, 1.0, 1.0)
        box_2 = make_box(1.0, 1.0, 1.0)

        box_2.translate((5.0, 5.0, 5.0))

        geometry = GeometryData(names=["a", "b"], geometrys=[box_1, box_2])

        overlaps = KDTreeOverlapDetector.detect(geometry)
        assert overlaps == []


class TestSpatialGridOverlapDetector:
    def test_build_grid(self):
        centres = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.9, 0.9, 0.9],
                [1.2, 1.2, 1.2],
            ]
        )

        grid = SpatialGridOverlapDetector.build_grid(centres, cell_size=1.0)

        assert grid[(0, 0, 0)] == [0, 1]
        assert grid[(1, 1, 1)] == [2]

    def test_grid_pairs(self):
        grid = {
            (0, 0, 0): [0, 1],
            (1, 0, 0): [2],
        }

        pairs = SpatialGridOverlapDetector.grid_pairs(grid)

        expected = {
            (0, 1),
            (0, 2),
            (1, 2),
        }

        assert {tuple(pair) for pair in pairs} == expected

    def test_detect_finds_overlap(self):
        box_1 = make_box(1.0, 1.0, 1.0)
        box_2 = make_box(1.0, 1.0, 1.0)

        box_2.translate((0.5, 0.5, 0.5))

        geometry = GeometryData(names=["a", "b"], geometrys=[box_1, box_2])

        overlaps = SpatialGridOverlapDetector.detect(geometry)
        assert overlaps == [("a", "b")]
