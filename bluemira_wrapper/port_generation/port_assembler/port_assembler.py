from __future__ import annotations

from typing import get_type_hints

from eudemo.comp_managers import ThermalShield
from eudemo.vacuum_vessel import VacuumVessel

from bluemira.base.reactor import ComponentManager, Reactor

from .ts_port_builders import (
    TSEquatorialPortBuilder,
    TSLowerPortBuilder,
    TSUpperPortBuilder,
)
from .vv_port_builders import (
    VVEquatorialPortBuilder,
    VVLowerPortBuilder,
    VVUpperPortBuilder,
)


class ReactorPortAssembler:
    """Build ports, reassign managers back onto reactor, return reactor."""

    @staticmethod
    def add_upper_ports(
        reactor: Reactor,
        params,
        build_config,
    ) -> Reactor:
        name_manager_pairs = ReactorPortAssembler._get_named_managers(reactor)

        for name, manager in name_manager_pairs:
            if isinstance(manager, ThermalShield):
                manager = TSUpperPortBuilder(params, build_config, manager).build()
            elif isinstance(manager, VacuumVessel):
                manager = VVUpperPortBuilder(params, build_config, manager).build()

            setattr(reactor, name, manager)

        return reactor

    @staticmethod
    def add_equatorial_ports(
        reactor: Reactor,
        params,
        build_config,
    ) -> Reactor:
        name_manager_pairs = ReactorPortAssembler._get_named_managers(reactor)

        for name, manager in name_manager_pairs:
            if isinstance(manager, ThermalShield):
                manager = TSEquatorialPortBuilder(
                    params,
                    build_config,
                    manager,
                ).build()
            elif isinstance(manager, VacuumVessel):
                manager = VVEquatorialPortBuilder(
                    params,
                    build_config,
                    manager,
                ).build()

            setattr(reactor, name, manager)

        return reactor

    @staticmethod
    def add_lower_ports(
        reactor: Reactor,
        params,
        build_config,
    ) -> Reactor:
        name_manager_pairs = ReactorPortAssembler._get_named_managers(reactor)

        for name, manager in name_manager_pairs:
            if isinstance(manager, ThermalShield):
                manager = TSLowerPortBuilder(params, build_config, manager).build()
            elif isinstance(manager, VacuumVessel):
                manager = VVLowerPortBuilder(params, build_config, manager).build()

            setattr(reactor, name, manager)

        return reactor

    @staticmethod
    def _get_named_managers(
        reactor: Reactor,
    ) -> tuple[tuple[str, ComponentManager], ...]:
        """Return (name, manager) pairs from reactor."""

        return tuple(
            (name, getattr(reactor, name))
            for name, component_type in get_type_hints(type(reactor)).items()
            if issubclass(component_type, ComponentManager)
            and getattr(reactor, name, None) is not None
        )
