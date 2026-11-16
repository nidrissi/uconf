"""Cocommutative cooperad: linear dual of the commutative operad.

``CoCommutative(n)`` is the arity-``n`` component of ``Com^*``, rank-one
in each arity ``n >= 1`` with zero differential.  The unique basis element
is ``()`` (empty tuple) and the infinitesimal cocomposition satisfies
``Δ^{i;m,n}(e_{m+n-1}) = e_m ⊗ e_n`` for all valid ``i``, dualising
``Com.compose(e_m, i, e_n) = e_{m+n-1}``.
"""

from __future__ import annotations

from sage.all import tensor

from uconf.core.cooperad import Cooperad, CooperadElement
from uconf.models.commutative import CommutativeComponent, CommutativeElement


class CoCommutativeElement(CommutativeElement, CooperadElement["CoCommutativeComponent"]):
    """Elements of a fixed-arity cocommutative cooperad component."""


class CoCommutativeComponent(CommutativeComponent):
    """Arity-``n`` component; the same graded module as ``Com(n)``."""

    Element = CoCommutativeElement


class CoCommutativeCooperad(Cooperad[CoCommutativeComponent]):
    """The cocommutative cooperad ``Com^*``; use the instance :data:`CoCommutative`.

    This is the linear dual of :data:`uconf.models.commutative.Commutative`.
    Its components are the same graded modules as those of ``Commutative``;
    the additional cooperad structure is provided by :meth:`counit`,
    :meth:`reduced`, and :meth:`infinitesimal_cocompose`.
    """

    Component = CoCommutativeComponent
    Element = CoCommutativeElement

    name = "coCom"
    connectivity = 0

    def unit_key(self) -> tuple:
        """Return the basis key of the counit generator in arity ``1``."""
        return ()

    def counit(self, x: CoCommutativeElement):
        """Cooperadic counit: extracts the coefficient of the arity-1 generator.

        The counit ``ε: CoCom(1) → k`` evaluates to 1 on the unique basis
        element ``()`` and to 0 for all other arities.
        """
        if x.arity() != 1:
            return x.parent().base_ring().zero()
        for _, coeff in x:
            return coeff
        return x.parent().base_ring().zero()

    def reduced(self, x: CoCommutativeElement) -> CoCommutativeElement:
        """Project to the reduced part (kills the arity-1 unit component)."""
        if x.arity() != 1:
            return x
        return x - self.counit(x) * x.parent()(())

    def infinitesimal_cocompose(self, x: CoCommutativeElement, i: int, m: int, n: int):
        """Partial cocomposition dual to ``Commutative.compose(·, i, ·)``.

        Since ``Com.compose(e_m, i, e_n) = e_{m+n-1}`` for all ``i``, the
        dual satisfies ``Δ^{i;m,n}(e_{m+n-1}) = e_m ⊗ e_n``.
        """
        if m <= 0 or n <= 0:
            raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")
        if x.arity() != m + n - 1:
            raise ValueError(f"Expected element in arity {m + n - 1}, got arity {x.arity()}.")

        base_ring = x.parent().base_ring()
        left_parent = self(m, base_ring=base_ring)
        right_parent = self(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        result = target.zero()
        for _, coeff in x:
            result += coeff * left_parent(()).tensor(right_parent(()))
        return result


CoCommutative = CoCommutativeCooperad()
