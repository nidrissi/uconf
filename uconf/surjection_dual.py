"""Linear dual model associated to the surjection operad."""

from __future__ import annotations

import itertools

from sage.all import tensor

from .surjection import Surjection


class SurjectionDual(Surjection):
    """Linear dual companion of :class:`uconf.surjection.Surjection`."""

    name = "S*"

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

    def _boundary_on_basis(self, basis_element: tuple) -> "SurjectionDual.Element":
        """Compute the differential by transposing ``Surjection`` differential."""

        primal_parent = Surjection(self.arity(), base_ring=self.base_ring())
        primal_degree = len(basis_element) - self.arity()
        source_degree = primal_degree + 1

        def term_generator():
            for source_term in primal_parent.basis_it(source_degree):
                source_basis = next(iter(source_term.support()))
                source_boundary = primal_parent._boundary_on_basis(source_basis)
                coeff = self.base_ring().zero()
                for target_basis, c in source_boundary:
                    if target_basis == basis_element:
                        coeff += c
                if coeff != 0:
                    yield (source_basis, coeff)

        return self.sum_of_terms(term_generator())

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
            out = target.zero()

            for left_degree in range(u_degree + 1):
                right_degree = u_degree - left_degree
                for left_term in left_parent.basis_it(-left_degree):
                    left_basis = next(iter(left_term.support()))
                    for right_term in right_parent.basis_it(-right_degree):
                        right_basis = next(iter(right_term.support()))
                        composed = Surjection.compose(left_term, i, right_term)
                        coeff = source_parent.base_ring().zero()
                        for basis, c in composed:
                            if basis == u_basis:
                                coeff += c
                        if coeff != 0:
                            out += coeff * (
                                left_parent.term(left_basis).tensor(
                                    right_parent.term(right_basis)
                                )
                            )
            return out

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
