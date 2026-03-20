"""Typing protocols for cooperad-like objects used in this project.

.. note::
   This package works exclusively with **connected** cooperads, i.e. cooperads C
   satisfying C(0) = 0 and C(1) = k (the ground field, spanned by the counit).
   Connectedness implies that every internal vertex of a tree in the cobar
   construction has arity >= 2, which bounds the number of vertices in arity n
   by n - 1.  All concrete models and wrappers in this package are connected.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, TypeAlias, runtime_checkable

from uconf.core.component import ComponentProtocol


@runtime_checkable
class CooperadComponent(ComponentProtocol, Protocol):
    """Structural contract for one fixed-arity cooperad component."""

    factory: CooperadLike
    """Reference to the parent cooperad factory, used for printing and error
    messages."""

    @staticmethod
    def counit(x: CooperadComponent.Element) -> object:
        """Returns the counit evaluation on ``x``."""
        ...

    @staticmethod
    def unit_key() -> Any:
        """Returns the basis key of the counit generator in arity ``1``."""
        ...

    @staticmethod
    def reduced(x: CooperadComponent.Element) -> CooperadComponent.Element:
        """Projects ``x`` to the reduced part (kills counit in arity 1)."""
        ...

    @staticmethod
    def infinitesimal_cocompose(
        x: CooperadComponent.Element, i: int, m: int, n: int
    ) -> Iterable[tuple]:
        """Returns the partial cocomposition in slot ``i``, arities ``(m, n)``."""
        ...


@runtime_checkable
class CooperadFactory(Protocol):
    """Structural contract for cooperad factories and wrappers."""

    name: str

    def __call__(self, n: int, base_ring) -> Component:
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def counit(self, x: Any) -> object:
        """Returns the counit evaluation on ``x``."""
        ...

    def unit_key(self) -> Any:
        """Returns the basis key of the counit generator in arity ``1``."""
        ...

    def reduced(self, x: Any) -> Any:
        """Projects ``x`` to the reduced part."""
        ...

    def infinitesimal_cocompose(
        self,
        x: Any,
        i: int,
        m: int,
        n: int,
    ) -> Any:
        """Returns the partial cocomposition in slot ``i``."""
        ...

    class Component(CooperadComponent): ...


CooperadLike: TypeAlias = type[CooperadComponent] | CooperadFactory
"""Type alias for accepted cooperad inputs (class or factory instance)."""
