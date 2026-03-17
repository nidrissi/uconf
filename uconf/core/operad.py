"""Typing protocols for operad-like objects used in this project.

.. note::
   This package works exclusively with **connected** operads, i.e. operads P
   satisfying P(0) = 0 and P(1) = k (the ground field, spanned by the unit).
   Connectedness implies that every internal vertex of a tree in the bar
   construction has arity >= 2, which bounds the number of vertices in arity n
   by n - 1.  All concrete models and wrappers in this package are connected.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

from uconf.core.component import ComponentProtocol


@runtime_checkable
class OperadComponent(ComponentProtocol, Protocol):
    """Structural contract for one fixed-arity operad component."""

    @staticmethod
    def compose(
        x: OperadComponent.Element, input: int, y: OperadComponent.Element
    ) -> OperadComponent.Element:
        """Computes ``x \\circ_i y``."""
        ...

    @staticmethod
    def unit(base_ring: Any) -> OperadComponent.Element:
        """Returns the unit element of the operad."""
        ...


@runtime_checkable
class OperadFactory(Protocol):
    """Structural contract for operad factories and wrappers.

    Factory instances (for example shifted/Hadamard wrappers) are callable by
    arity and expose operadic structure maps at the factory level.
    """

    name: str

    def __call__(self, n: int, base_ring) -> Component:
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def unit(self, base_ring) -> Any:
        """Returns the unit element of the operad."""
        ...

    def compose(
        self,
        x: Any,
        i: int,
        y: Any,
    ) -> Any:
        """Computes ``x \\circ_i y``."""
        ...

    class Component(OperadComponent): ...


OperadLike: TypeAlias = type[OperadComponent] | OperadFactory
"""Type alias for accepted operad inputs (class or factory instance)."""
