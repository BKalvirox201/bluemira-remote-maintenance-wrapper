from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from bluemira.base.parameter_frame import Parameter
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, make_polygon

from bluemira_wrapper.blanket_grouping import group_blanket_components
from bluemira_wrapper.validators.blanket_validators.blanket_validator import (
    BlanketValidationParameters,
    BlanketValidator,
    BlanketValidatorError,
    ValidationResult,
)

BLANKET_VALIDATOR_PATH = (
    "bluemira_wrapper.validators.blanket_validators.blanket_validator"
)


@pytest.fixture
def physical_component(material):
    shape = Mock()
    shape.volume = 2.0
    shape.center_of_mass = np.array([0.5, 0.5, 0.5])

    shape.bounding_box = SimpleNamespace(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
    )

    component = Mock()
    component.name = "OBS_FW_0 1"
    component.shape = shape
    component.material = material

    return component


def make_physical_component(shape, name: str = "OBS_FW_0 1"):
    component = Mock()
    component.name = name
    component.shape = shape
    component.material = Mock()

    return component


def group_parts(*parts):
    return group_blanket_components(parts)


@pytest.fixture
def material():
    material = Mock()
    material.density.return_value = 1000.0
    return material


@pytest.fixture
def validation_params():
    return BlanketValidationParameters(
        bb_mass_limit=Parameter("bb_mass_limit", 5000.0),
        bb_max_width=Parameter("bb_max_width", 5.0),
        bb_max_length=Parameter("bb_max_length", 5.0),
        bb_max_height=Parameter("bb_max_height", 5.0),
        bb_upper_port_blanket_clearance=Parameter(
            "bb_upper_port_blanket_clearance", 0.1
        ),
    )


@pytest.fixture
def validator(blanket, validation_params):
    return BlanketValidator(blanket, validation_params)


@pytest.fixture
def blanket():
    return Mock()


class TestMassWithinLimits:
    def test_passes_when_mass_below_limit(
        self, blanket, validation_params, physical_component
    ):
        physical_component.name = "OBS_FW_0 1"
        validator = BlanketValidator(blanket, validation_params)
        result = validator._mass_within_limits(
            group_parts(physical_component),
            mass_limit=5000.0,
        )

        assert result == ValidationResult("_mass_within_limits", True, "")

    def test_fails_when_mass_exceeds_limit(
        self, blanket, validation_params, physical_component
    ):
        physical_component.name = "OBS_FW_0 1"
        validator = BlanketValidator(blanket, validation_params)
        result = validator._mass_within_limits(
            group_parts(physical_component),
            mass_limit=1000.0,
        )

        assert result.name == "_mass_within_limits"
        assert not result.passed
        assert "OBS blanket 0" in result.message

    def test_checks_all_parts_when_mass_exceeds_limit(
        self, blanket, validation_params
    ):
        parts = (
            make_physical_component(SimpleNamespace(volume=2.0), name="OBS_FW_0 1"),
            make_physical_component(SimpleNamespace(volume=0.5), name="OBS_BZ_0 1"),
            make_physical_component(
                SimpleNamespace(volume=3.0), name="OBS_MANIFOLD_0 1"
            ),
            make_physical_component(SimpleNamespace(volume=0.5), name="OBS_FW_1 1"),
        )
        for part in parts:
            part.material.density.return_value = 1000.0

        validator = BlanketValidator(blanket, validation_params)
        result = validator._mass_within_limits(
            group_parts(*parts), mass_limit=1000.0
        )

        assert not result.passed
        assert "OBS blanket 0" in result.message
        assert "OBS blanket 1" not in result.message
        for part in parts:
            part.material.density.assert_called_once()

    def test_fails_when_density_unavailable(
        self, blanket, validation_params, physical_component
    ):
        physical_component.name = "OBS_FW_0 1"
        physical_component.material.density.side_effect = NotImplementedError
        validator = BlanketValidator(blanket, validation_params)
        result = validator._mass_within_limits(
            group_parts(physical_component),
            mass_limit=1000.0,
        )

        assert result.name == "_mass_within_limits"
        assert not result.passed
        assert "density" in result.message


class TestGeometryDimensionChecks:
    @patch(BLANKET_VALIDATOR_PATH + ".normalise_part_rotation")
    def test_dimensions_within_limits_passes(
        self, normalise_mock, blanket, validation_params, physical_component
    ):
        normalise_mock.return_value = physical_component.shape
        physical_component.name = "OBS_FW_0 1"
        validator = BlanketValidator(blanket, validation_params)
        result = validator._dimensions_within_limits(
            group_parts(physical_component),
            maximum_length=2.0,
            maximum_width=2.0,
            maximum_height=2.0,
        )

        assert result == ValidationResult("_dimensions_within_limits", True, "")

    @patch(BLANKET_VALIDATOR_PATH + ".normalise_part_rotation")
    def test_dimensions_fail_when_height_exceeds_limit(
        self, normalise_mock, blanket, validation_params, physical_component
    ):
        normalise_mock.return_value = physical_component.shape
        physical_component.name = "OBS_FW_0 1"
        physical_component.shape.bounding_box.z_max = 3.0
        validator = BlanketValidator(blanket, validation_params)
        result = validator._dimensions_within_limits(
            group_parts(physical_component),
            maximum_length=2.0,
            maximum_width=2.0,
            maximum_height=2.0,
        )

        assert not result.passed
        assert "maximum height" in result.message

    @patch(BLANKET_VALIDATOR_PATH + ".normalise_part_rotation")
    def test_dimensions_include_all_components_in_blanket_group(
        self, normalise_mock, blanket, validation_params
    ):
        parts = (
            make_physical_component(
                SimpleNamespace(
                    bounding_box=SimpleNamespace(
                        x_min=0.0,
                        x_max=1.0,
                        y_min=0.0,
                        y_max=1.0,
                        z_min=0.0,
                        z_max=1.0,
                    )
                ),
                name="OBS_FW_0 1",
            ),
            make_physical_component(
                SimpleNamespace(
                    bounding_box=SimpleNamespace(
                        x_min=1.0,
                        x_max=2.0,
                        y_min=0.0,
                        y_max=1.0,
                        z_min=0.0,
                        z_max=1.0,
                    )
                ),
                name="OBS_BZ_0 1",
            ),
        )
        normalise_mock.side_effect = lambda shape: shape

        validator = BlanketValidator(blanket, validation_params)
        result = validator._dimensions_within_limits(
            group_parts(*parts),
            maximum_length=1.5,
            maximum_width=2.0,
            maximum_height=2.0,
        )

        assert not result.passed
        assert "OBS blanket 0" in result.message
        assert "maximum length" in result.message


class TestAllPartsFitUpperPort:
    @staticmethod
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

    @staticmethod
    def make_box(width: float, depth: float, height: float):
        return extrude_shape(
            TestAllPartsFitUpperPort.make_rectangular_face(width, depth),
            (0.0, 0.0, height),
        )

    def test_small_part_fits_large_opening(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=2.0, height=2.0)
        part = make_physical_component(
            self.make_rectangular_face(width=1.0, height=1.0), name="OBS_FW_0 1"
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert results[0].passed

    def test_large_part_does_not_fit_small_opening(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=1.0, height=1.0)
        part = make_physical_component(
            self.make_rectangular_face(width=2.0, height=2.0), name="OBS_FW_0 1"
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert not results[0].passed

    def test_exact_same_size_does_not_fit(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=1.0, height=1.0)
        part = make_physical_component(
            self.make_rectangular_face(width=1.0, height=1.0), name="OBS_FW_0 1"
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert not results[0].passed

    def test_translation_independent(self, validator: BlanketValidator):
        shape = self.make_rectangular_face(width=1.0, height=1.0)
        shape.translate((100.0, 50.0, 0.0))
        opening = self.make_rectangular_face(width=2.0, height=2.0)
        opening.translate((-75.0, 25.0, 0.0))
        part = make_physical_component(shape)
        part.name = "OBS_FW_0 1"
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert results[0].passed

    def test_box_projection_fits(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=2.0, height=2.0)
        part = make_physical_component(
            self.make_box(width=1.0, depth=1.0, height=10.0), name="OBS_FW_0 1"
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert results[0].passed

    def test_multiple_parts(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=2.0, height=2.0)
        fitting_part = make_physical_component(
            self.make_rectangular_face(width=1.0, height=1.0),
            name="OBS_FW_0 1",
        )
        oversized_part = make_physical_component(
            self.make_rectangular_face(width=3.0, height=3.0),
            name="IBS_FW_0 1",
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(fitting_part, oversized_part), opening, clearance_m=0.0
        )

        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed

    def test_grouped_parts_must_fit_together(self, validator: BlanketValidator):
        opening = self.make_rectangular_face(width=2.0, height=2.0)
        first_part = make_physical_component(
            self.make_rectangular_face(width=1.0, height=1.0),
            name="OBS_FW_0 1",
        )
        second_shape = self.make_rectangular_face(width=1.0, height=1.0)
        second_shape.translate((1.5, 0.0, 0.0))
        second_part = make_physical_component(second_shape, name="OBS_BZ_0 1")

        results = validator._all_parts_fit_upper_port(
            group_parts(first_part, second_part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert not results[0].passed
        assert "OBS blanket 0" in results[0].message

    def test_numerical_boundary_tolerance_allows_fit(
        self, validator: BlanketValidator
    ):
        opening = self.make_rectangular_face(width=2.0, height=1.5)
        part = make_physical_component(
            self.make_rectangular_face(width=1.4, height=1.4), name="OBS_FW_0 1"
        )
        results = validator._all_parts_fit_upper_port(
            group_parts(part), opening, clearance_m=0.0
        )

        assert len(results) == 1
        assert results[0].passed
