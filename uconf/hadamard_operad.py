"""Hadamard product wrapper for operads.

For operads ``P`` and ``Q``, this module implements the aritywise tensor-product
operad ``P ⊙ Q``:

- ``(P ⊙ Q)(n) = P(n) ⊗ Q(n)`` as graded modules,
- differential ``d(p ⊗ q) = dp ⊗ q + (-1)^|p| p ⊗ dq``,
- symmetric action diagonal: ``(p ⊗ q)·σ = (p·σ) ⊗ (q·σ)``,
- partial composition diagonal:
  ``(p ⊗ q) ∘_i (p' ⊗ q') = (p ∘_i p') ⊗ (q ∘_i q')``.
"""

from __future__ import annotations

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, QQ, SymmetricGroup

from .operad import OperadProtocol


class HadamardProduct:
    """Factory for Hadamard-product operad components."""

    def __init__(
        self,
        left_operad_cls: type[OperadProtocol],
        right_operad_cls: type[OperadProtocol],
    ):
        self.left_operad_cls = left_operad_cls
        self.right_operad_cls = right_operad_cls
        self.name = f"{left_operad_cls.name}⊙{right_operad_cls.name}"

    def __call__(self, n: int, base_ring=QQ) -> "HadamardProduct.Component":
        return HadamardProduct.Component(self, n, base_ring)

    def unit(self, base_ring=QQ) -> "HadamardProduct.Element":
        component = self(1, base_ring)
        return component.from_factors(
            self.left_operad_cls.unit(),
            self.right_operad_cls.unit(),
        )

    def compose(
        self,
        x: "HadamardProduct.Element",
        i: int,
        y: "HadamardProduct.Element",
    ) -> "HadamardProduct.Element":
        x_parent = x.parent()
        y_parent = y.parent()

        if x_parent.factory is not self or y_parent.factory is not self:
            raise TypeError("Both elements must belong to this Hadamard operad.")
        if x_parent.base_ring() != y_parent.base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x_parent.arity()
        n = y_parent.arity()
        target = self(m + n - 1, x_parent.base_ring())

        left_x = x_parent.left_parent()
        left_y = y_parent.left_parent()
        right_x = x_parent.right_parent()
        right_y = y_parent.right_parent()

        accumulated = target.zero()
        for x_basis, x_coeff in x:
            left_x_basis, right_x_basis = x_basis
            left_x_term = left_x.term(left_x_basis)
            right_x_term = right_x.term(right_x_basis)
            for y_basis, y_coeff in y:
                left_y_basis, right_y_basis = y_basis
                left_y_term = left_y.term(left_y_basis)
                right_y_term = right_y.term(right_y_basis)

                left_composed = self.left_operad_cls.compose(
                    left_x_term, i, left_y_term
                )
                right_composed = self.right_operad_cls.compose(
                    right_x_term, i, right_y_term
                )

                for left_basis, left_coeff in left_composed:
                    for right_basis, right_coeff in right_composed:
                        accumulated += target.term((left_basis, right_basis)) * (
                            x_coeff * y_coeff * left_coeff * right_coeff
                        )

        return accumulated

    class Component(CombinatorialFreeModule):
        """A fixed-arity component of ``P ⊙ Q``."""

        def __init__(self, factory: "HadamardProduct", n: int, base_ring=QQ):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            name = f"{factory.name}{n}"
            super().__init__(
                base_ring,
                tuple,
                prefix=name,
                category=GradedModulesWithBasis(base_ring),
            )
            self.factory = factory
            self._arity = int(n)
            self._left_parent = factory.left_operad_cls(n, base_ring)
            self._right_parent = factory.right_operad_cls(n, base_ring)
            self._symmetric_group = SymmetricGroup(self._arity)
            self.rename(name)
            self.boundary = self.module_morphism(
                on_basis=self._boundary_on_basis,
                codomain=self,
            )

        def _validate_basis_key(self, basis_key):
            if not isinstance(basis_key, (tuple, list)) or len(basis_key) != 2:
                raise TypeError("Basis key must be a pair (left_basis, right_basis).")
            left_basis, right_basis = basis_key

            if hasattr(self._left_parent, "_validate_basis_key"):
                left_basis = self._left_parent._validate_basis_key(left_basis)
            if hasattr(self._right_parent, "_validate_basis_key"):
                right_basis = self._right_parent._validate_basis_key(right_basis)

            if left_basis is None or right_basis is None:
                return None
            return (left_basis, right_basis)

        def _element_constructor_(self, x):
            if isinstance(x, HadamardProduct.Element):
                return self.sum_of_terms((basis, coeff) for basis, coeff in x)

            if isinstance(x, dict):
                clean_dict = {}
                for key, coeff in x.items():
                    clean_key = self._validate_basis_key(key)
                    if clean_key is None:
                        continue
                    clean_dict[clean_key] = coeff
                return super()._element_constructor_(clean_dict)

            if isinstance(x, (tuple, list)):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                return self.term(clean_key)

            return super()._element_constructor_(x)

        def _boundary_on_basis(self, basis_element):
            left_basis, right_basis = basis_element
            left_term = self._left_parent.term(left_basis)
            right_term = self._right_parent.term(right_basis)

            left_degree = self._left_parent.degree_on_basis(left_basis)
            left_boundary = self._left_parent.boundary(left_term)
            right_boundary = self._right_parent.boundary(right_term)

            sign = -1 if left_degree % 2 else 1
            result = self.zero()

            for new_left_basis, left_coeff in left_boundary:
                result += self.term((new_left_basis, right_basis)) * left_coeff
            for new_right_basis, right_coeff in right_boundary:
                result += self.term((left_basis, new_right_basis)) * (
                    sign * right_coeff
                )

            return result

        def arity(self) -> int:
            return self._arity

        def left_parent(self):
            return self._left_parent

        def right_parent(self):
            return self._right_parent

        def base_ring(self):
            return self._left_parent.base_ring()

        def degree_on_basis(self, basis_element) -> int:
            left_basis, right_basis = basis_element
            return self._left_parent.degree_on_basis(
                left_basis
            ) + self._right_parent.degree_on_basis(right_basis)

        def from_factors(
            self, left_element, right_element
        ) -> "HadamardProduct.Element":
            if left_element.parent().arity() != self.arity():
                raise TypeError(
                    f"Left element arity {left_element.parent().arity()} does not match target arity {self.arity()}."
                )
            if right_element.parent().arity() != self.arity():
                raise TypeError(
                    f"Right element arity {right_element.parent().arity()} does not match target arity {self.arity()}."
                )
            if left_element.parent().base_ring() != self.base_ring():
                raise TypeError("Left element must have the same base ring as target.")
            if right_element.parent().base_ring() != self.base_ring():
                raise TypeError("Right element must have the same base ring as target.")

            return self.sum_of_terms(
                (
                    (left_basis, right_basis),
                    left_coeff * right_coeff,
                )
                for left_basis, left_coeff in left_element
                for right_basis, right_coeff in right_element
            )

        def compose(
            self,
            x: "HadamardProduct.Element",
            i: int,
            y: "HadamardProduct.Element",
        ) -> "HadamardProduct.Element":
            return self.factory.compose(x, i, y)

        def unit(self) -> "HadamardProduct.Element":
            return self.factory.unit(self.base_ring())

    class Element(CombinatorialFreeModule.Element):
        """Element wrapper carrying Hadamard-operad structure maps."""

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "HadamardProduct.Element":
            return self.parent().boundary(self)

        def permute(self, sigma) -> "HadamardProduct.Element":
            if isinstance(sigma, (list, tuple)):
                sigma = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity()}. Got {sigma} ({type(sigma)})."
                )

            result = self.parent().zero()
            for basis, coeff in self:
                left_basis, right_basis = basis
                left_term = self.parent().left_parent().term(left_basis).permute(sigma)
                right_term = (
                    self.parent().right_parent().term(right_basis).permute(sigma)
                )
                result += self.parent().from_factors(left_term, right_term) * coeff
            return result


HadamardProduct.Component.Element = HadamardProduct.Element
