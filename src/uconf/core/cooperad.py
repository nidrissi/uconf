"""Base class for cooperads.

.. note::
   This package works exclusively with **connected** cooperads, i.e. cooperads C
   satisfying C(0) = 0 and C(1) = k (the ground field, spanned by the counit).
   Connectedness implies that every internal vertex of a tree in the cobar
   construction has arity >= 2, which bounds the number of vertices in arity n
   by n - 1.  All concrete models and wrappers in this package are connected.
"""

from __future__ import annotations

from typing import Any, Self, cast

from uconf.core.component import ComponentBase, ElementBase, SymmetricSequence


class Cooperad[C: ComponentBase[Any]](SymmetricSequence[C]):
    """A connected dg-cooperad, given by its components and structure maps.

    Every cooperad in this package is an instance of a subclass, e.g.
    ``CoAssociative``, ``ShiftedCooperad(SurjectionDual, 1)`` or
    ``BarConstruction(Lie)``.
    """

    def counit(self, x: Any) -> Any:
        """Return the counit evaluation of ``x``, an element of the base ring."""
        raise NotImplementedError

    def reduced(self, x: Any) -> Any:
        """Project ``x`` to the reduced part (kills the counit in arity ``1``)."""
        raise NotImplementedError

    def infinitesimal_cocompose(self, x: Any, i: int, m: int, n: int) -> Any:
        """Return the partial cocomposition of ``x`` in slot ``i``, arities ``(m, n)``.

        The result lives in ``C(m) ⊗ C(n)``.
        """
        raise NotImplementedError


class CooperadElement[P: ComponentBase[Any]](ElementBase[P]):
    """Element of a cooperad component, with the structure maps as methods."""

    def _cooperad(self) -> Cooperad[Any]:
        return cast(Cooperad[Any], self.parent().factory)

    def counit(self) -> Any:
        """Return the counit evaluation of this element."""
        return self._cooperad().counit(self)

    def reduced(self) -> Self:
        """Project to the reduced part."""
        return self._cooperad().reduced(self)

    def infinitesimal_cocompose(self, i: int, m: int, n: int) -> Any:
        """Return the partial cocomposition in slot ``i``, arities ``(m, n)``."""
        return self._cooperad().infinitesimal_cocompose(self, i, m, n)
