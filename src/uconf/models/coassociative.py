"""Coassociative cooperad: linear dual of the associative operad.

``CoAssociative(n)`` is the arity-``n`` component of ``Ass^*``, spanned by
the dual permutation basis with zero differential.  The infinitesimal
cocomposition ``Δ^{i;m,n}`` is the transpose of ``Associative.compose(·, i, ·)``.
"""

from __future__ import annotations

import itertools

from sage.all import tensor

from uconf.core.cooperad import Cooperad, CooperadElement
from uconf.models.associative import Associative, AssociativeComponent, AssociativeElement


class CoAssociativeElement(AssociativeElement, CooperadElement["CoAssociativeComponent"]):
    """Elements of a fixed-arity coassociative cooperad component."""


class CoAssociativeComponent(AssociativeComponent):
    """Arity-``n`` component; the same graded module as ``Ass(n)``."""

    Element = CoAssociativeElement


class CoAssociativeCooperad(Cooperad[CoAssociativeComponent]):
    """The coassociative cooperad ``Ass^*``; use the instance :data:`CoAssociative`.

    This is the linear dual of :data:`uconf.models.associative.Associative`.
    Its components are the same graded modules as those of ``Associative``;
    the additional cooperad structure is provided by :meth:`counit`,
    :meth:`reduced`, and :meth:`infinitesimal_cocompose`.
    """

    Component = CoAssociativeComponent
    Element = CoAssociativeElement

    name = "coAss"
    connectivity = 0

    def unit_key(self) -> tuple:
        """Return the basis key of the counit generator in arity ``1``."""
        return (1,)

    def counit(self, x: CoAssociativeElement):
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

    def reduced(self, x: CoAssociativeElement) -> CoAssociativeElement:
        """Project to the reduced part (kills the arity-1 unit component)."""
        if x.arity() != 1:
            return x
        return x - self.counit(x) * x.parent()((1,))

    def infinitesimal_cocompose(self, x: CoAssociativeElement, i: int, m: int, n: int):
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
            raise ValueError(f"Expected element in arity {m + n - 1}, got arity {x.arity()}.")

        base_ring = x.parent().base_ring()
        left_parent = self(m, base_ring=base_ring)
        right_parent = self(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        def _on_basis(sigma: tuple) -> object:
            out = target.zero()
            for left_key in itertools.permutations(range(1, m + 1)):
                for right_key in itertools.permutations(range(1, n + 1)):
                    composed = Associative.compose(
                        left_parent(left_key), i, right_parent(right_key)
                    )
                    for composed_basis, composed_coeff in composed:
                        if composed_basis == sigma:
                            out += composed_coeff * left_parent(left_key).tensor(
                                right_parent(right_key)
                            )
            return out

        result = target.zero()
        for sigma_basis, sigma_coeff in x:
            result += sigma_coeff * _on_basis(sigma_basis)
        return result


CoAssociative = CoAssociativeCooperad()
