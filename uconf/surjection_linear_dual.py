"""Linear dual model associated to the surjection operad."""

from __future__ import annotations

from sage.all import tensor

from .surjection import Surjection


class SurjectionLinearDual(Surjection):
    """Linear dual companion of :class:`uconf.surjection.Surjection`."""

    name = "S*"

    @staticmethod
    def counit(x: "SurjectionLinearDual.Element"):
        """Return the cooperadic counit evaluation."""

        if x.arity() != 1:
            return x.parent().base_ring().zero()
        value = x.parent().base_ring().zero()
        for basis, coeff in x:
            if basis == (1,):
                value += coeff
        return value

    @staticmethod
    def reduced(x: "SurjectionLinearDual.Element") -> "SurjectionLinearDual.Element":
        """Project to the reduced part (kills the arity-1 unit component)."""

        if x.arity() != 1:
            return x
        return x - SurjectionLinearDual.counit(x) * x.parent()((1,))

    @staticmethod
    def infinitesimal_cocompose(
        x: "SurjectionLinearDual.Element", i: int, m: int, n: int
    ):
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
        left_parent = SurjectionLinearDual(m, base_ring=base_ring)
        right_parent = SurjectionLinearDual(n, base_ring=base_ring)
        target = tensor([left_parent, right_parent])

        if not x:
            return target.zero()

        source_parent = x.parent()

        def _on_basis(u_basis: tuple[int, ...]):
            u_degree = source_parent.degree_on_basis(u_basis)
            out = target.zero()

            for left_degree in range(u_degree + 1):
                right_degree = u_degree - left_degree
                for left_term in left_parent.basis_it(left_degree):
                    left_basis = next(iter(left_term.support()))
                    for right_term in right_parent.basis_it(right_degree):
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

            return SurjectionLinearDual.counit(self)

        def reduced(self) -> "SurjectionLinearDual.Element":
            """Project to the reduced part."""

            return SurjectionLinearDual.reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Return the partial cocomposition dual to ``compose``."""

            return SurjectionLinearDual.infinitesimal_cocompose(self, i, m, n)
