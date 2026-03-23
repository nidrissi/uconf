"""Utilities for typed Sage element parent access."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, cast


ParentT = TypeVar("ParentT")


class ParentedElementMixin(Generic[ParentT]):
    """Mixin providing typed ``parent()`` helpers.

    Sage element classes expose ``parent()`` dynamically. This mixin centralizes
    the static typing bridge so subclasses can avoid repeating local casts.
    """

    def parent(self) -> ParentT:
        return cast(ParentT, cast(Any, super()).parent())
