"""Coassociative cooperad: linear dual of the associative operad.

``CoAssociative(n)`` is the arity-``n`` component of ``Ass^*``, spanned by
the dual permutation basis with zero differential.  The infinitesimal
cocomposition ``Δ^{i;m,n}`` is the transpose of ``Associative.compose(·, i, ·)``.
"""

from __future__ import annotations

import itertools
from typing import ClassVar

from sage.all import QQ, tensor

from uconf.models.associative import Associative


class CoAssociative(Associative):
    """Coassociative cooperad component in fixed arity.

    This is the linear dual of :class:`uconf.associative.Associative`.
    As a graded module it is identical to ``Associative``; the additional
    cooperad structure is provided by :meth:`counit`, :meth:`reduced`, and
    :meth:`infinitesimal_cocompose`.
    """

    name: ClassVar[str] = "coAss"

    @staticmethod
    def counit(x: "CoAssociative.Element"):
        """Cooperadic counit: extracts coefficient of the arity-1 generator.

        The counit ``ε: CoAss(1) → k`` evaluates to 1 on the unique basis
        element ``(1,)`` and to 0 for all other arities.
        """
        if x.arity() != 1:
            return x.parent().base_ring().zero()
        value = x.parent().base_ring().zero()
        for basis, coeff in x:
            if basis == (1,):
                value += coeff
        return value

    @staticmethod
    def reduced(x: "CoAssociative.Element") -> "CoAssociative.Element":
        """Project to the reduced part (kills the arity-1 unit component)."""
        if x.arity() != 1:
            return x
        return x - CoAssociative.counit(x) * x.parent()((1,))

    @staticmethod
    def infinitesimal_cocompose(x: "CoAssociative.Element", i: int, m: int, n: int):
        """Partial cocomposition dual to ``Associative.compose(·, i, ·)``.

        For each basis element ``sigma ∈ S_{m+n-1}`` in ``x``, returns
        ``Σ tau ⊗ rho`` summed over all ``(tau, rho) ∈ S_m × S_n`` with
        ``Ass.compose(tau, i, rho) == sigma``.
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
        left_parent = CoAssociative(m, base_ring=base_ring)
        right_parent = CoAssociative(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        def _on_basis(sigma: tuple) -> object:
            out = target.zero()
            for left_key in itertools.permutations(range(1, m + 1)):
                for right_key in itertools.permutations(range(1, n + 1)):
                    composed = Associative.compose(
                        left_parent.term(left_key), i, right_parent.term(right_key)
                    )
                    for composed_basis, composed_coeff in composed:
                        if composed_basis == sigma:
                            out += composed_coeff * left_parent.term(left_key).tensor(
                                right_parent.term(right_key)
                            )
            return out

        result = target.zero()
        for sigma_basis, sigma_coeff in x:
            result += sigma_coeff * _on_basis(sigma_basis)
        return result

    class Element(Associative.Element):
        """Elements of a fixed-arity coassociative cooperad component."""

        def counit(self):
            """Return the cooperadic counit evaluation."""
            return CoAssociative.counit(self)

        def reduced(self) -> "CoAssociative.Element":
            """Project to the reduced part."""
            return CoAssociative.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Return the partial cocomposition dual to ``Associative.compose``."""
            return CoAssociative.infinitesimal_cocompose(self, i, m, n)
