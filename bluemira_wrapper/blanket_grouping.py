# SPDX-FileCopyrightText: 2026-present N.Salt
#
# SPDX-License-Identifier: MIT 2026

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable


class _NamedComponent(Protocol):
    name: str


_BLANKET_COMPONENT_NAME = re.compile(r"^(IBS|OBS)_[^_]+_(\d+)\s+(\d+)$")


def group_blanket_components[T: _NamedComponent](
    components: Iterable[T],
) -> dict[tuple[str, int], list[T]]:
    grouped_components: dict[tuple[str, int], list[T]] = defaultdict(list)

    for component in components:
        match = _BLANKET_COMPONENT_NAME.match(component.name)
        if match is None:
            raise ValueError(f"Unexpected blanket component name: {component.name}")

        group_key = (match.group(1), int(match.group(2)))
        grouped_components[group_key].append(component)

    return dict(grouped_components)
