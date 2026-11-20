from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import (
    Parameter,
    ParameterFrame,
    make_parameter_frame,
)
from bluemira.geometry.base import BluemiraGeo
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import offset_wire
from matproplib.conditions import STPConditions
from shapely import Polygon, affinity
from shapely.geometry.base import BaseGeometry

from bluemira_wrapper.blanket_grouping import group_blanket_components
from eudemo.blanket import Blanket


class ValidationError(Exception): ...


@dataclass(slots=True)
class ValidationResult:
    name: str
    passed: bool
    message: str = ""


class Validator(abc.ABC): ...


class BlanketValidatorError(ValidationError):
    pass


@dataclass
class BlanketValidationParameters(ParameterFrame):
    bb_max_width: Parameter[float]
    bb_max_length: Parameter[float]
    bb_max_height: Parameter[float]
    bb_upper_port_blanket_clearance: Parameter[float]
    bb_mass_limit: Parameter[float]


class BlanketValidator(Validator):
    param_cls: type[BlanketValidationParameters] = BlanketValidationParameters
    params: BlanketValidationParameters

    def __init__(self, blanket: Blanket, params: dict | ParameterFrame):
        self.blanket = blanket
        _params = make_parameter_frame(params, self.param_cls)
        assert _params
        self.params = _params

    def execute(self, upper_port_void_face: BluemiraFace) -> list[ValidationResult]:
        """_summary_

        Args:
            upper_port_void_face (BluemiraFace): _description_

        Raises:
            BlanketValidatorError: _description_

        Returns:
            list[ValidationResult]: _description_
        """
        blanket_parts = _get_all_blanket_subcomponents(self.blanket)
        blanket_groups = group_blanket_components(blanket_parts)

        results: list[ValidationResult] = [
            self._dimensions_within_limits(
                blanket_groups,
                maximum_length=self.params.bb_max_length.value,
                maximum_width=self.params.bb_max_width.value,
                maximum_height=self.params.bb_max_height.value,
            ),
            self._mass_within_limits(blanket_groups, self.params.bb_mass_limit.value),
        ]
        results.extend(
            self._all_parts_fit_upper_port(
                blanket_groups,
                upper_port_void_face,
                self.params.bb_upper_port_blanket_clearance.value,
            ),
        )

        failures = [result for result in results if not result.passed]
        if failures:
            raise BlanketValidatorError(
                "\n".join(failure.message for failure in failures)
            )

        return results

    def _dimensions_within_limits(
        self,
        blanket_groups: dict[tuple[str, int], list[PhysicalComponent]],
        maximum_length: float,
        maximum_width: float,
        maximum_height: float,
    ) -> ValidationResult:
        failure_messages: list[str] = []

        for group_key, parts in blanket_groups.items():
            # TODO: Consider normalising rotation in y as well.
            bounding_boxes = [
                normalise_part_rotation(part.shape).bounding_box for part in parts
            ]
            length = max(box.x_max for box in bounding_boxes) - min(
                box.x_min for box in bounding_boxes
            )
            width = max(box.y_max for box in bounding_boxes) - min(
                box.y_min for box in bounding_boxes
            )
            height = max(box.z_max for box in bounding_boxes) - min(
                box.z_min for box in bounding_boxes
            )
            group_name = _blanket_group_name(group_key)

            if length > maximum_length:
                failure_messages.append(
                    f"{group_name} exceeds maximum length. "
                    f"Length={length:.3f} m "
                    f"Limit={maximum_length:.3f} m"
                )
            if width > maximum_width:
                failure_messages.append(
                    f"{group_name} exceeds maximum width. "
                    f"Width={width:.3f} m "
                    f"Limit={maximum_width:.3f} m"
                )
            if height > maximum_height:
                failure_messages.append(
                    f"{group_name} exceeds maximum height. "
                    f"Height={height:.3f} m "
                    f"Limit={maximum_height:.3f} m"
                )

        return ValidationResult(
            self._dimensions_within_limits.__name__,
            len(failure_messages) == 0,
            "\n".join(failure_messages),
        )

    def _all_parts_fit_upper_port(
        self,
        blanket_groups: dict[tuple[str, int], list[PhysicalComponent]],
        upper_port_face: BluemiraFace,
        clearance_m: float,
    ) -> list[ValidationResult]:
        """
        Checks that all blanket parts can fit through the upper port

        Args:
            blanket_parts (tuple[PhysicalComponent, ...]): _description_
            upper_port_face (BluemiraFace): _description_
            clearance_m (float): _description_
            rotation_step_degrees (int, optional): _description_. Defaults to 1.

        Returns:
            list[ValidationResult]: _description_
        """
        results: list[ValidationResult] = []
        exterior_wire = upper_port_face.wires[0]
        clearance_wire = offset_wire(
            exterior_wire,
            -clearance_m if clearance_m > 0.001 else -0.001,
            open_wire=False,
        )
        silhouette = BluemiraFace(clearance_wire)
        normalised_silhouette = normalise_port_rotation(silhouette)
        opening_points = points_from_vertices(normalised_silhouette.vertexes)
        opening_convex_hull = Polygon(opening_points[:, :2]).convex_hull

        for group_key, parts in blanket_groups.items():
            part_points = np.vstack([
                points_from_vertices(normalise_part_rotation(part.shape).vertexes)
                for part in parts
            ])
            part_convex_hull = Polygon(part_points[:, :2]).convex_hull
            centred_part_convex_hull = affinity.translate(
                part_convex_hull,
                xoff=opening_convex_hull.bounds[2] - part_convex_hull.bounds[2],
                yoff=opening_convex_hull.centroid.y - part_convex_hull.centroid.y,
            )
            group_name = _blanket_group_name(group_key)
            plot_polygons(opening_convex_hull, centred_part_convex_hull, group_name)

            if opening_convex_hull.buffer(1e-9).covers(centred_part_convex_hull):
                results.append(
                    ValidationResult(self._all_parts_fit_upper_port.__name__, True, "")
                )
            else:
                results.append(
                    ValidationResult(
                        self._all_parts_fit_upper_port.__name__,
                        False,
                        (
                            f"{group_name} cannot fit through "
                            f"upper port with clearance {clearance_m:.3f} m"
                        ),
                    )
                )

        return results

    def _mass_within_limits(
        self,
        blanket_groups: dict[tuple[str, int], list[PhysicalComponent]],
        mass_limit: float,
    ) -> ValidationResult:
        failure_messages: list[str] = []
        for group_key, parts in blanket_groups.items():
            mass = 0.0
            for part in parts:
                try:
                    if not part.material:
                        raise AttributeError
                    density = float(part.material.density(STPConditions()))

                except (AttributeError, TypeError, ValueError, NotImplementedError):
                    return ValidationResult(
                        self._mass_within_limits.__name__,
                        False,
                        f"Blanket {_blanket_group_name(group_key)} component "
                        f"{part.name} missing valid density value",
                    )

                mass += density * part.shape.volume

            if mass > mass_limit:
                failure_messages.append(
                    f"{_blanket_group_name(group_key)} exceeds mass limit. "
                    f"Mass={mass:.3f} Limit={mass_limit:.3f}"
                )

        return ValidationResult(
            self._mass_within_limits.__name__,
            not failure_messages,
            "\n".join(failure_messages),
        )


def _blanket_group_name(group_key: tuple[str, int]) -> str:
    return f"{group_key[0]} blanket {group_key[1]}"


def _get_all_blanket_subcomponents(
    blanket: Blanket,
) -> tuple[PhysicalComponent, ...]:
    blanket_component: Component = blanket.component()
    # NOTE: if we just get the xy vertexes. It just gives a crossection at z=0
    blanket_xyz = cast("Component", blanket_component.get_component("xyz"))
    sector_one_blanket = cast("Component", blanket_xyz.get_component("BB 1"))
    return tuple(
        cast("PhysicalComponent", child) for child in sector_one_blanket.children
    )


def points_from_vertices(vertices) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[1] == 3:
        return points
    return points.T


def normalise_part_rotation(part: BluemiraGeo) -> BluemiraGeo:
    points = points_from_vertices(part.vertexes)[:, :2]
    polygon = Polygon(points)

    rectangle = polygon.minimum_rotated_rectangle
    rectangle_points = np.asarray(rectangle.boundary.coords[:-1])

    edge_vectors = np.diff(np.vstack([rectangle_points, rectangle_points[0]]), axis=0)
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    longest_edge = edge_vectors[np.argmax(edge_lengths)]
    angle = -np.degrees(np.arctan2(longest_edge[1], longest_edge[0]))
    normalised_polygon = affinity.rotate(polygon, angle, origin="centroid")

    # NOTE: We have to then point the shortest edge of the trapezoid at the origin
    normalised_points = np.asarray(normalised_polygon.exterior.coords[:-1])
    x_values = np.round(normalised_points[:, 0], 6)
    left_side = normalised_points[x_values == x_values.min()]
    right_side = normalised_points[x_values == x_values.max()]
    left_length = np.ptp(left_side[:, 1])
    right_length = np.ptp(right_side[:, 1])
    left_distance = abs(x_values.min())
    right_distance = abs(x_values.max())

    needs_flip = (left_length > right_length and left_distance < right_distance) or (
        right_length > left_length and right_distance < left_distance
    )
    final_angle = angle + (180.0 if needs_flip else 0.0)
    part.rotate(tuple(part.center_of_mass.tolist()), (0.0, 0.0, 1.0), final_angle)
    return part


def normalise_port_rotation(part: BluemiraFace) -> BluemiraFace:
    vertices = points_from_vertices(part.vertexes)
    points_2d = vertices[:, :2]

    if len(points_2d) < 3:
        raise ValueError("Expected at least 3 vertices.")

    edge_vectors = np.diff(np.vstack([points_2d, points_2d[0]]), axis=0)

    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    shortest_edge = edge_vectors[np.argmin(edge_lengths)]
    angle = np.degrees(np.arctan2(shortest_edge[1], shortest_edge[0]))
    rotation_angle = 90.0 - angle

    part.rotate(
        base=tuple(part.center_of_mass.tolist()),
        direction=(0, 0, 1),
        degree=rotation_angle,
    )

    return part


def plot_polygon(part_geometry: BaseGeometry, title: str = "") -> None:
    figure, axis = plt.subplots()

    part_x, part_y = part_geometry.boundary.xy
    axis.plot(part_x, part_y, "r-", linewidth=2, label="Part")
    axis.fill(part_x, part_y, alpha=0.2, color="red")

    axis.set_aspect("equal")
    axis.legend()
    axis.set_title(title)

    plt.show()


def plot_polygons(
    opening_geometry: BaseGeometry, part_geometry: BaseGeometry, title: str = ""
) -> None:
    figure, axis = plt.subplots()

    opening_x, opening_y = opening_geometry.boundary.xy
    axis.plot(opening_x, opening_y, "b-", linewidth=2, label="Opening")
    axis.fill(opening_x, opening_y, alpha=0.2, color="blue")

    part_x, part_y = part_geometry.boundary.xy
    axis.plot(part_x, part_y, "r-", linewidth=2, label="Part")
    axis.fill(part_x, part_y, alpha=0.2, color="red")

    axis.set_aspect("equal")
    axis.legend()
    axis.set_title(title)

    plt.show()
