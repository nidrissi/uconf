"""Linear dual model associated to the surjection operad."""

from __future__ import annotations

import itertools
from typing import ClassVar

from sage.all import cached_method, tensor

from uconf.models.surjection import Surjection


class SurjectionDual(Surjection):
    """Linear dual companion of :class:`uconf.surjection.Surjection`."""

    name: ClassVar[str] = "S*"
    connectivity: ClassVar[int] = 0
    """Elements live in non-positive degrees; the minimum degree is n - (n+d) = -d <= 0
    for primal degree d >= 0.  We use k = 0 as the connectivity convention (inherited
    from the primal surjection operad) since the cobar construction uses the arity bound
    n - 1 for the weight, independent of the sign of the degree."""

    def __init__(self, n: int, base_ring):
        """Initialize fixed-arity linear dual component."""

        super().__init__(n, base_ring=base_ring)
        self._primal_parent = Surjection(n, base_ring=self.base_ring())

    def basis_it(self, d: int):
        """Iterate over basis elements in dual degree ``d`` (non-positive)."""

        assert d <= 0, f"d must be a non-positive integer, got d={d}."
        r = self.arity()
        for values in itertools.product(range(1, r + 1), repeat=r - d):
            res = self(values)
            if res != self.zero():
                yield res

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Return dual homological degree (negative primal degree)."""

        return self.arity() - len(basis_element)

    @cached_method
    def _boundary_rows_for_degree(
        self, primal_degree: int
    ) -> dict[tuple[int, ...], dict]:
        """Return the transposed boundary "matrix" for a fixed primal degree.

        For each ``source_basis`` of primal degree ``primal_degree + 1``,
        iterates over its primal boundary and accumulates, for every
        ``target_basis``, the coefficient ``<target_basis*, ∂(source_basis)>``.
        The result maps each ``target_basis`` to the dict
        ``{source_basis: coeff}`` representing the row of ``∂^T``.
        """

        source_degree = primal_degree + 1
        rows: dict[tuple[int, ...], dict] = {}
        for source_term in self._primal_parent.basis_it(source_degree):
            source_basis = next(iter(source_term.support()))
            source_boundary = self._primal_parent._boundary_on_basis(source_basis)
            for target_basis, coeff in source_boundary:
                if target_basis not in rows:
                    rows[target_basis] = {}
                rows[target_basis][source_basis] = (
                    rows[target_basis].get(source_basis, self.base_ring().zero())
                    + coeff
                )
        return rows

    def _boundary_on_basis(self, basis_element: tuple) -> "SurjectionDual.Element":
        """Compute the differential by transposing ``Surjection`` differential."""

        primal_degree = len(basis_element) - self.arity()
        row = self._boundary_rows_for_degree(primal_degree).get(basis_element, {})
        return self.sum_of_terms(row.items())

    @cached_method
    def _cocompose_rows(self, i: int, m: int, n: int, u_degree: int) -> dict:
        """Return cached pairing rows for ``Δ^{i;m,n}`` in fixed primal degree."""

        left_parent = SurjectionDual(m, base_ring=self.base_ring())
        right_parent = SurjectionDual(n, base_ring=self.base_ring())
        rows: dict[tuple[int, ...], dict] = {}

        for left_degree in range(u_degree + 1):
            right_degree = u_degree - left_degree

            left_basis_terms = [
                (next(iter(left_term.support())), left_term)
                for left_term in left_parent.basis_it(-left_degree)
            ]
            right_basis_terms = [
                (next(iter(right_term.support())), right_term)
                for right_term in right_parent.basis_it(-right_degree)
            ]

            for left_basis, left_term in left_basis_terms:
                for right_basis, right_term in right_basis_terms:
                    composed = Surjection.compose(left_term, i, right_term)
                    pair_basis = (left_basis, right_basis)
                    for u_basis, coeff in composed:
                        if u_basis not in rows:
                            rows[u_basis] = {}
                        rows[u_basis][pair_basis] = (
                            rows[u_basis].get(pair_basis, self.base_ring().zero())
                            + coeff
                        )

        return rows

    @staticmethod
    def counit(x: "SurjectionDual.Element"):
        """Return the cooperadic counit evaluation."""

        if x.arity() != 1:
            return x.parent().base_ring().zero()
        value = x.parent().base_ring().zero()
        for basis, coeff in x:
            if basis == (1,):
                value += coeff
        return value

    @staticmethod
    def reduced(x: "SurjectionDual.Element") -> "SurjectionDual.Element":
        """Project to the reduced part (kills the arity-1 unit component)."""

        if x.arity() != 1:
            return x
        return x - SurjectionDual.counit(x) * x.parent()((1,))

    @staticmethod
    def infinitesimal_cocompose(x: "SurjectionDual.Element", i: int, m: int, n: int):
        """Partial cocomposition dual to ``Surjection.compose`` on basis pairing."""

        if m <= 0 or n <= 0:
            raise ValueError(f"Arities must be positive. Got m={m}, n={n}.")
        if not (1 <= i <= m):
            raise ValueError(f"Index i must satisfy 1 <= i <= {m}. Got i={i}.")
        if x.arity() != m + n - 1:
            raise ValueError(
                f"Expected element in arity {m + n - 1}, got arity {x.arity()}."
            )

        base_ring = x.parent().base_ring()
        left_parent = SurjectionDual(m, base_ring=base_ring)
        right_parent = SurjectionDual(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        source_parent = x.parent()

        def _on_basis(u_basis: tuple[int, ...]):
            u_degree = -source_parent.degree_on_basis(u_basis)
            rows = source_parent._cocompose_rows(i, m, n, u_degree)
            return target.sum_of_terms(rows.get(u_basis, {}).items())

        result = target.zero()
        for u_basis, u_coeff in x:
            result += u_coeff * _on_basis(u_basis)
        return result

    class Element(Surjection.Element):
        """Elements of a fixed-arity surjection linear-dual component."""

        def counit(self):
            """Return the cooperadic counit evaluation."""

            return SurjectionDual.counit(self)

        def reduced(self) -> "SurjectionDual.Element":
            """Project to the reduced part."""

            return SurjectionDual.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Return the partial cocomposition dual to ``compose``."""

            return SurjectionDual.infinitesimal_cocompose(self, i, m, n)


SurjectionDual.factory = SurjectionDual
