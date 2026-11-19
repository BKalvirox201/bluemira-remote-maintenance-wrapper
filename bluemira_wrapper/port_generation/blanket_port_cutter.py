from __future__ import annotations

from typing import cast

from bluemira_st.blanket.manager import BB
from port_builder import PortBuilder

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.tools import boolean_cut


class BlanketPortCutter(PortBuilder):
    def __init__(self, params, build_config):
        self.params = params
        self.build_config = build_config

    def cut_equatorial_ports(self, manager: BB) -> BB:
        shape, material = self._get_shape_and_material(manager)

        x_max = shape.bounding_box.x_max + 0.1

        offset = (
            self.params.tk_vv_single_wall.value
            + self.params.g_vv_bb.value
            + self.params.tk_bb.value
        )

        yz_face, yz_void = self._build_port_faces(
            x_max,
            self.params.ep_width.value,
            self.params.ep_height.value,
            self.params.ep_z_position.value,
            self.params.tk_bb.value,
            offset,
        )

        degree = 180 / self.params.n_TF.value
        vector = (self.params.R_0.value - x_max, 0, 0)

        _, void = self._extrude_rotate_port(yz_face, yz_void, vector, degree)
        cut_shape = boolean_cut(shape, void)
        assert not isinstance(cut_shape, list), "Expected single geometry"
        new_component = PhysicalComponent("BB cut", cut_shape, material)

        root = Component("BB eq port cut")
        new_component.parent = root

        return cast(BB, self._replace_xyz_geometry(manager, root))

    def cut_upper_ports(self, manager: ComponentManager) -> Component:
        raise NotImplementedError

    def cut_lower_ports(self, manager: ComponentManager) -> Component:
        raise NotImplementedError
