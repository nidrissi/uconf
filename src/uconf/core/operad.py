"""Base class for operads.

.. note::
   This package works exclusively with **connected** operads, i.e. operads P
   satisfying P(0) = 0 and P(1) = k (the ground field, spanned by the unit).
   Connectedness implies that every internal vertex of a tree in the bar
   construction has arity >= 2, which bounds the number of vertices in arity n
   by n - 1.  All concrete models and wrappers in this package are connected.
"""

from __future__ import annotations

from typing import Any

from uconf.core.component import ComponentBase, SymmetricSequence


class Operad[C: ComponentBase[Any]](SymmetricSequence[C]):
    """A connected dg-operad, given by its components and structure maps.

    Every operad in this package is an instance of a subclass, e.g. ``Lie``,
    ``ShiftedOperad(Lie, 1)`` or ``CobarConstruction(CoAssociative)``.
    """

    def compose(self, x: Any, i: int, y: Any, /) -> Any:
        """Compute the partial composition ``x \\circ_i y``."""
        raise NotImplementedError

    def unit(self, base_ring: Any) -> Any:
        """Return the operadic unit, an element of the arity-``1`` component."""
        raise NotImplementedError
