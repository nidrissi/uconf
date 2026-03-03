"""Cocommutative cooperad: linear dual of the commutative operad.

``CoCommutative(n)`` is the arity-``n`` component of ``Com^*``, rank-one
in each arity ``n >= 1`` with zero differential.  The unique basis element
is ``()`` (empty tuple) and the infinitesimal cocomposition satisfies
``Δ^{i;m,n}(e_{m+n-1}) = e_m ⊗ e_n`` for all valid ``i``, dualising
``Com.compose(e_m, i, e_n) = e_{m+n-1}``.
"""

from __future__ import annotations

from typing import ClassVar

from sage.all import QQ, tensor

from .commutative import Commutative


class CoCommutative(Commutative):
    """Cocommutative cooperad component in fixed arity.

    This is the linear dual of :class:`uconf.commutative.Commutative`.
    As a graded module it is identical to ``Commutative``; the additional
    cooperad structure is provided by :meth:`counit`, :meth:`reduced`, and
    :meth:`infinitesimal_cocompose`.
    """

    name: ClassVar[str] = "coCom"

    @staticmethod
    def counit(x: "CoCommutative.Element"):
        """Cooperadic counit: extracts the coefficient of the arity-1 generator.

        The counit ``ε: CoCom(1) → k`` evaluates to 1 on the unique basis
        element ``()`` and to 0 for all other arities.
        """
        if x.arity() != 1:
            return x.parent().base_ring().zero()
        for _, coeff in x:
            return coeff
        return x.parent().base_ring().zero()

    @staticmethod
    def reduced(x: "CoCommutative.Element") -> "CoCommutative.Element":
        """Project to the reduced part (kills the arity-1 unit component)."""
        if x.arity() != 1:
            return x
        return x - CoCommutative.counit(x) * x.parent()(())

    @staticmethod
    def infinitesimal_cocompose(
        x: "CoCommutative.Element", i: int, m: int, n: int
    ):
        """Partial cocomposition dual to ``Commutative.compose(·, i, ·)``.

        Since ``Com.compose(e_m, i, e_n) = e_{m+n-1}`` for all ``i``, the
        dual satisfies ``Δ^{i;m,n}(e_{m+n-1}) = e_m ⊗ e_n``.
        """
        if m <= 0 or n <= 0:
            raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")
        if x.arity() != m + n - 1:
            raise ValueError(
                f"Expected element in arity {m + n - 1}, got arity {x.arity()}."
            )

        base_ring = x.parent().base_ring()
        left_parent = CoCommutative(m, base_ring=base_ring)
        right_parent = CoCommutative(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        result = target.zero()
        for _, coeff in x:
            result += coeff * left_parent.term(()).tensor(right_parent.term(()))
        return result

    class Element(Commutative.Element):
        """Elements of a fixed-arity cocommutative cooperad component."""

        def counit(self):
            """Return the cooperadic counit evaluation."""
            return CoCommutative.counit(self)

        def reduced(self) -> "CoCommutative.Element":
            """Project to the reduced part."""
            return CoCommutative.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Return the partial cocomposition dual to ``Commutative.compose``."""
            return CoCommutative.infinitesimal_cocompose(self, i, m, n)
