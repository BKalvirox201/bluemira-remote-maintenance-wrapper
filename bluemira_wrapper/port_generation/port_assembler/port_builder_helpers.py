from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from matproplib.material import Material

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraGeo
from bluemira.geometry.tools import (
    boolean_cut,
    convert,
    extrude_shape,
    make_polygon,
    offset_wire,
)

if TYPE_CHECKING:
    from Part import Shape  # type: ignore

PartShape: TypeAlias = "Shape"


class PortBuilderError(Exception):
    """Raised when port builder geometry cannot be created."""


def cut_vessel(shape, void):
    result = boolean_cut(shape, [void])
    if isinstance(result, list):
        return max(result, key=lambda s: s.volume)

    if isinstance(result, list):
        print([s.volume for s in result])
        print("orig:", shape.volume)
    else:
        print("single result", result.volume)

    return result


def build_output(name, vessel_shape, port_solid, material):
    root = Component(f"{name} assembly")
    xyz = Component("xyz")

    vessel = PhysicalComponent("vessel", vessel_shape, material)
    port = PhysicalComponent(f"{name} port", port_solid, material)

    vessel.parent = xyz
    port.parent = xyz
    xyz.parent = root

    return root


def make_equatorial_port_yz_face(
    x_ref: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    wall_side_thickness: float,
) -> BluemiraFace:

    # expand void slightly → avoids grazing edge
    padding = max(0.2, wall_side_thickness * 2)

    inner = make_polygon(
        {
            "x": [x_ref] * 4,
            "y": [y_min - padding, y_min - padding, y_max + padding, y_max + padding],
            "z": [z_min - padding, z_max + padding, z_max + padding, z_min - padding],
        },
        closed=True,
    )

    # grow outer more aggressively → guarantees volume
    outer = offset_wire(
        inner,
        wall_side_thickness * 2.0,
        open_wire=False,
    )

    return BluemiraFace([outer, inner])


def get_shape_and_material(
    manager: ComponentManager,
) -> tuple[BluemiraGeo, Material | None]:
    component: Component = manager.component()
    xyz_node = component.get_component("xyz")

    def find_physical(node):
        for child in getattr(node, "children", []):
            if isinstance(child, PhysicalComponent):
                return child
            result = find_physical(child)
            if result:
                return result
        return None

    physical = find_physical(xyz_node)
    if physical is None:
        raise PortBuilderError(f"No PhysicalComponent found under xyz tree")
    return physical.shape, physical.material


def replace_xyz_geometry(manager, new_root, target, body_key):
    root = manager.component()

    def find_component(node, name):
        if node.name == name:
            return node
        for child in node.children:
            found = find_component(child, name)
            if found:
                return found
        return None

    target_component = find_component(root, target)
    xyz = target_component.get_component("xyz")
    sector = next((c for c in xyz.children if c.name.startswith("Sector")), None)
    new_xyz = new_root.get_component("xyz")
    new_vessel = next(
        c
        for c in new_xyz.children
        if isinstance(c, PhysicalComponent) and "port" not in c.name.lower()
    )

    updated = []

    for child in list(sector.children):
        print(child.name.lower())

        child.parent = None
        if body_key in child.name.lower():
            new_vessel.name = child.name
            updated.append(new_vessel)
        else:
            updated.append(child)

    sector.children = []

    for c in updated:
        c.parent = sector

    return type(manager)(root)


def attach_port(manager, target, port):
    if port is None:
        return manager

    root = manager.component()

    def find_component(node, name):
        if node.name == name:
            return node
        for child in node.children:
            found = find_component(child, name)
            if found:
                return found
        return None

    comp = find_component(root, target)
    xyz = comp.get_component("xyz")
    sector = next(c for c in xyz.children if c.name.startswith("Sector"))

    names = {c.name for c in sector.children}

    if port.name not in names:
        port.parent = None  # ✅ detach
        port.parent = sector

    return type(manager)(root)


def extract_from_root(root):
    xyz = root.get_component("xyz")

    vessel = None
    port = None

    for c in xyz.children:
        if isinstance(c, PhysicalComponent):
            if "port" in c.name.lower():
                port = c
            else:
                vessel = c

    return vessel, port


def compute_port_centre(shape):
    bbox = shape.bounding_box
    return 0.5 * (bbox.x_min + bbox.x_max)


def build_port_xy_face_from_centre(
    number_of_tf_coils,
    centre_x,
    centre_y,
    port_width,
    wall_end_thickness,
    wall_side_thickness,
):
    half_width = 0.5 * port_width

    x_min = centre_x - half_width
    x_max = centre_x + half_width

    half_sector_angle = np.pi / number_of_tf_coils
    cosine_half_angle = np.cos(half_sector_angle)
    tangent_half_angle = np.tan(half_sector_angle)

    outer_y = centre_y / cosine_half_angle
    inner_y = outer_y + wall_side_thickness / cosine_half_angle

    x_outer_min = x_min

    a_coeff = 1 + tangent_half_angle**2
    b_coeff = -2 * outer_y * tangent_half_angle
    c_coeff = outer_y**2 - x_max**2

    x_outer_max = (
        0.5 * (-b_coeff + np.sqrt(b_coeff**2 - 4 * a_coeff * c_coeff)) / a_coeff
    )

    x_inner_min = x_outer_min + wall_end_thickness
    x_inner_max = x_outer_max - wall_end_thickness

    y_outer_min = x_outer_min * tangent_half_angle - outer_y
    if y_outer_min < 0:
        y_outer_min = 0
        x_outer_min = outer_y / tangent_half_angle
        x_inner_min = x_outer_min + wall_end_thickness

    y_inner_min = x_inner_min * tangent_half_angle - inner_y
    y_inner_max = x_inner_max * tangent_half_angle - inner_y

    if y_inner_min < 0:
        y_inner_min = 0
        c_adjusted = y_inner_max - tangent_half_angle * x_inner_max
        x_inner_min = -c_adjusted / tangent_half_angle
        x_outer_min = x_inner_min - wall_end_thickness

    y_outer_max = x_outer_max * tangent_half_angle - outer_y

    inner_wire = make_polygon(
        {
            "x": [x_inner_min, x_inner_max, x_inner_max, x_inner_min],
            "y": [-y_inner_min, -y_inner_max, y_inner_max, y_inner_min],
        },
        closed=True,
    )

    outer_wire = make_polygon(
        {
            "x": [x_outer_min, x_outer_max, x_outer_max, x_outer_min],
            "y": [-y_outer_min, -y_outer_max, y_outer_max, y_outer_min],
        },
        closed=True,
    )

    port_face = BluemiraFace([outer_wire, inner_wire])
    port_face.rotate(degree=np.rad2deg(half_sector_angle))

    return port_face


def describe_geometry(label: str, geometry):
    try:
        if isinstance(geometry, list):
            print(f"{label}: list len={len(geometry)}")
            for index, solid in enumerate(geometry):
                print(
                    f"  [{index}] volume={getattr(solid, 'volume', None)} "
                    f"z_max={getattr(solid.bounding_box, 'z_max', None)}"
                )
        else:
            print(
                f"{label}: volume={getattr(geometry, 'volume', None)} "
                f"z_max={getattr(geometry.bounding_box, 'z_max', None)}"
            )
    except Exception as error:
        print(f"{label}: describe failed -> {error}")


def build_upper_port_from_extrude(
    profile_face: BluemiraFace,
    void_face: BluemiraFace,
    outer_vessel: BluemiraGeo,
    inner_vessel: BluemiraGeo,
    direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> BluemiraGeo:
    """Extrude profile, intersect outer vessel, remove inner vessel, cut void, return port solid."""

    if profile_face is None:
        raise RuntimeError("profile_face invalid")
    if void_face is None:
        raise RuntimeError("void_face invalid")
    if outer_vessel is None or inner_vessel is None:
        raise RuntimeError("vessel invalid")

    dx, dy, dz = direction_vector
    if dx == 0 and dy == 0 and dz == 0:
        raise RuntimeError("direction zero")

    length = 10000.0
    extrusion_vector = (dx * length, dy * length, dz * length)

    extruded_port = extrude_shape(profile_face, extrusion_vector)
    describe_geometry("extruded_port", extruded_port)
    print(f"profile face bounding box: {profile_face.bounding_box}")
    print(f"extruded port bounding box: {extruded_port.bounding_box}")
    extruded_void = extrude_shape(void_face, extrusion_vector)
    describe_geometry("extruded_void", extruded_void)

    raw = outer_vessel.shape.common(extruded_port.shape)
    describe_geometry("raw intersection", raw)

    if hasattr(raw, "Solids") and raw.Solids:
        solids = [convert(s) for s in raw.Solids]
    else:
        solids = [convert(raw)]

    print(f"solids count after common: {len(solids)}")
    for index, solid in enumerate(solids):
        print(f"  [{index}] volume={solid.volume}")

    outer_solid = max(solids, key=lambda s: s.volume)
    describe_geometry("outer_solid", outer_solid)

    wall_region = boolean_cut(outer_solid, inner_vessel)
    describe_geometry("wall_region", wall_region)

    if wall_region is None:
        raise RuntimeError("inner cut failed")

    solids_after_wall_cut = (
        wall_region if isinstance(wall_region, list) else [wall_region]
    )
    print(f"solids_after_wall_cut count: {len(solids_after_wall_cut)}")

    if not solids_after_wall_cut:
        raise RuntimeError("no solids after wall cut")

    port_body = max(solids_after_wall_cut, key=lambda solid: solid.bounding_box.z_max)
    describe_geometry("port_body", port_body)

    port_with_void = boolean_cut(port_body, extruded_void)
    describe_geometry("port_with_void", port_with_void)

    if port_with_void is None:
        raise RuntimeError("void cut failed")

    solids_after_void_cut = (
        port_with_void if isinstance(port_with_void, list) else [port_with_void]
    )
    print(f"solids_after_void_cut count: {len(solids_after_void_cut)}")

    final_port = max(solids_after_void_cut, key=lambda solid: solid.volume)
    describe_geometry("final_port", final_port)

    return final_port
