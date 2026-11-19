from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from eudemo.vacuum_vessel import VacuumVessel

from bluemira.base.builder import Builder
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import extrude_shape, offset_wire, revolve_shape

from . import port_builder_helpers


@dataclass
class VVUpperPortDuctBuilderParams(ParameterFrame):
    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_vv_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    tk_vv_double_wall: Parameter[float]
    tk_vv_single_wall: Parameter[float]


@dataclass
class VVEquatorialPortDuctBuilderParams(ParameterFrame):
    R_0: Parameter[float]
    n_TF: Parameter[int]
    g_cr_ts: Parameter[float]
    ep_z_position: Parameter[float]
    ep_width: Parameter[float]
    ep_height: Parameter[float]
    tk_vv_single_wall: Parameter[float]


@dataclass
class VVLowerPortDuctBuilderParams(ParameterFrame):
    n_TF: Parameter[int]
    lower_port_angle: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    lp_width: Parameter[float]
    lp_height: Parameter[float]
    lp_duct_div_pad_ib: Parameter[float]
    lp_duct_div_pad_ob: Parameter[float]


class VVUpperPortBuilder(Builder):
    param_cls = VVUpperPortDuctBuilderParams

    def __init__(self, parameters, build_configuration, component_manager):
        super().__init__(parameters, build_configuration)
        self.component_manager: VacuumVessel = component_manager
        self.params: VVUpperPortDuctBuilderParams

    def build(self):
        vessel_shape, vessel_material = port_builder_helpers.get_shape_and_material(
            self.component_manager
        )
        print("VV Component Manager Tree")
        self.component_manager.component().print_tree()

        external_vessel_surface = revolve_shape(
            BluemiraFace(self.component_manager.xz_boundary),
            degree=360.0,
        )

        internal_vessel_surface = revolve_shape(
            BluemiraFace(self.component_manager.xz_inner_boundary),
            degree=360.0,
        )

        port_centre_x = port_builder_helpers.compute_port_centre(vessel_shape)
        port_width = 10.5 - 5.85
        port_centre_y = (
            self.params.tf_wp_depth.value
            + self.params.g_ts_tf.value
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

        port_face = port_builder_helpers.build_port_xy_face_from_centre(
            self.params.n_TF.value,
            port_centre_x,
            port_centre_y,
            port_width,
            self.params.tk_vv_double_wall.value,
            self.params.tk_vv_single_wall.value,
        )

        port_void_face = BluemiraFace(port_face.boundary[1])
        direction = (0.0, 0.0, -1.0)
        extrude_length = -vessel_shape.bounding_box.z_max
        port_void_solid = extrude_shape(port_void_face, (0, 0, -extrude_length))

        port_solid = port_builder_helpers.build_upper_port_from_extrude(
            port_face,
            port_void_face,
            external_vessel_surface,
            internal_vessel_surface,
            direction,
        )

        vessel_with_hole = port_builder_helpers.cut_vessel(
            vessel_shape,
            port_void_solid,
        )

        root_component = port_builder_helpers.build_output(
            "vv upper",
            vessel_with_hole,
            port_solid,
            vessel_material,
        )

        _, port_component = port_builder_helpers.extract_from_root(root_component)
        updated_manager = port_builder_helpers.replace_xyz_geometry(
            self.component_manager,
            root_component,
            target="VacuumVessel",
            body_key="body",
        )
        manager_with_port = port_builder_helpers.attach_port(
            updated_manager,
            "VacuumVessel",
            port_component,
        )

        return manager_with_port


class VVEquatorialPortBuilder(Builder):
    param_cls = VVEquatorialPortDuctBuilderParams

    def __init__(self, params, build_config, manager):
        super().__init__(params, build_config)
        self.manager: VacuumVessel = manager
        self.params: VVEquatorialPortDuctBuilderParams

    def build(self):
        shape, material = port_builder_helpers.get_shape_and_material(self.manager)
        bbox = shape.bounding_box

        x_max = bbox.x_max + 0.5

        y_val = 0.5 * self.params.ep_width.value
        z_ref = self.params.ep_z_position.value
        z_val = 0.5 * self.params.ep_height.value

        yz_face = port_builder_helpers.make_equatorial_port_yz_face(
            x_max,
            -y_val,
            y_val,
            z_ref - z_val,
            z_ref + z_val,
            self.params.tk_vv_single_wall.value * 1.2,
        )

        yz_voidface = BluemiraFace(yz_face.boundary[1])
        vec = (self.params.R_0.value - x_max, 0, 0)

        solid = extrude_shape(yz_face, vec)
        void = extrude_shape(yz_voidface, vec)

        degree = 180 / self.params.n_TF.value
        solid.rotate(degree=degree)
        void.rotate(degree=degree)

        vessel_cut = port_builder_helpers.cut_vessel(shape, void)
        root = port_builder_helpers.build_output("vv eq", vessel_cut, solid, material)
        _, port = port_builder_helpers.extract_from_root(root)
        manager = port_builder_helpers.replace_xyz_geometry(
            self.manager, root, target="VacuumVessel", body_key="body"
        )
        manager_with_port = port_builder_helpers.attach_port(
            manager, "VacuumVessel", port
        )

        print("Old Tree")
        self.manager.component().print_tree()
        print("New Tree")
        root.print_tree()
        print("New Tree with Port")
        manager_with_port.component().print_tree()
        return manager_with_port


class VVLowerPortBuilder(Builder):
    param_cls = VVLowerPortDuctBuilderParams

    def __init__(self, params, build_config, manager):
        super().__init__(params, build_config)
        self.manager = manager

    def build(self):
        self.params: VVLowerPortDuctBuilderParams
        shape, material = port_builder_helpers.get_shape_and_material(self.manager)

        x = shape.bounding_box.x_max
        z = shape.bounding_box.z_min

        inner = port_builder_helpers.make_rect_yz(
            x,
            self.params.lp_width.value,
            self.params.lp_height.value,
            z,
        )

        outer = offset_wire(inner, self.params.tk_vv_single_wall.value, open_wire=False)

        face = BluemiraFace([outer, inner])
        void_face = BluemiraFace(inner)

        angle = np.deg2rad(self.params.lower_port_angle.value)
        length = 6.0

        vec = (-length, 0, -length * np.tan(angle))

        solid = extrude_shape(face, vec)
        void = extrude_shape(void_face, vec)

        solid.rotate(degree=180 / self.params.n_TF.value)
        void.rotate(degree=180 / self.params.n_TF.value)

        solid, void = port_builder_helpers.force_penetration(shape, solid, void)

        vessel_cut = port_builder_helpers.cut_vessel(shape, void)

        root = port_builder_helpers.build_output(
            "vv lower", vessel_cut, solid, material
        )

        return port_builder_helpers.replace_xyz_geometry(
            self.manager, root, target="VacuumVessel"
        )
