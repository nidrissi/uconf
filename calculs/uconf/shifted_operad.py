"""Shifted operad construction with standard Koszul/sign-representation twists.

For an operad ``P`` and integer shift ``d``, this module implements a wrapper
``ShiftedOperad(P, d)`` whose arity-``n`` component is shifted by ``d (n - 1)``.
The symmetric-group action and composition follow the usual suspension-style
sign rules from standard operad references.
"""

from __future__ import annotations

from typing import Any

from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    QQ,
    SymmetricGroup,
)
from .signs import (
    shifted_boundary_sign,
    shifted_operadic_compose_sign,
    shifted_permutation_sign,
)


class ShiftedOperad:
    """Factory for shifted operad components.

    Args:
        operad_cls: Base operad class (e.g. ``Lie`` or ``Surjection``).
        shift_degree: Integer ``d`` in the arity-dependent shift ``d (n-1)``.

    The construction uses the following conventions for homogeneous terms:

    - Symmetric action twist by ``sgn(sigma)^d``.
    - Composition twist
      ``(-1)^(d * ((i-1)(n-1) + (m-1)|y|))``
      with ``m = arity(x)``, ``n = arity(y)``, and ``|y|`` in the base operad.
    """

    def __init__(self, operad_cls, shift_degree: int):
        self.operad_cls = operad_cls
        self.shift_degree = int(shift_degree)
        self.name = f"{operad_cls.name}[{self.shift_degree}]"

    def __call__(self, n: int, base_ring=QQ) -> "ShiftedOperad.Component":
        return ShiftedOperad.Component(self, n, base_ring)

    def unit(self, base_ring=QQ) -> "ShiftedOperad.Element":
        component = self(1, base_ring)
        return component.from_base(self.operad_cls.unit())

    def compose(
        self, x: "ShiftedOperad.Element", i: int, y: "ShiftedOperad.Element"
    ) -> "ShiftedOperad.Element":
        x_parent = x.parent()
        y_parent = y.parent()

        if x_parent.factory is not self or y_parent.factory is not self:
            raise TypeError("Both elements must belong to this shifted operad.")
        if x_parent.base_ring() != y_parent.base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x_parent.arity()
        n = y_parent.arity()
        target = self(m + n - 1, x_parent.base_ring())
        accumulated = target.zero()

        for x_basis, x_coeff in x:
            x_term = x_parent.base_parent().term(x_basis)
            for y_basis, y_coeff in y:
                y_term = y_parent.base_parent().term(y_basis)
                y_degree = y_parent.base_parent().degree_on_basis(y_basis)
                base_composed = self.operad_cls.compose(x_term, i, y_term)
                sign = shifted_operadic_compose_sign(
                    self.shift_degree,
                    i,
                    m,
                    n,
                    y_degree,
                )
                accumulated += target.sum_of_terms(
                    (
                        basis,
                        sign * x_coeff * y_coeff * coeff,
                    )
                    for basis, coeff in base_composed
                )

        return target.from_base(accumulated)

    class Component(CombinatorialFreeModule):
        """A fixed-arity component of a shifted operad."""

        def __init__(self, factory: "ShiftedOperad", n: int, base_ring=QQ):
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
            self._base_parent = factory.operad_cls(n, base_ring)
            self._symmetric_group = SymmetricGroup(self._arity)
            self.rename(name)
            self.boundary = self.module_morphism(
                on_basis=self._boundary_on_basis,
                codomain=self,
            )

        def _validate_basis_key(self, basis_key):
            if hasattr(self._base_parent, "_validate_basis_key"):
                return self._base_parent._validate_basis_key(basis_key)
            return basis_key

        def _element_constructor_(self, x):
            if isinstance(x, ShiftedOperad.Element):
                return self.from_base(x.base_element())

            if hasattr(x, "parent") and x.parent() is self._base_parent:
                return self.from_base(x)

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
            sign = shifted_boundary_sign(self.factory.shift_degree)
            base_bdry = sign * self._base_parent.boundary(
                self._base_parent.term(basis_element)
            )
            return self.sum_of_terms((basis, coeff) for basis, coeff in base_bdry)

        def __call__(self, x) -> "ShiftedOperad.Element":
            return super().__call__(x)

        def arity(self) -> int:
            return self._arity

        def base_ring(self):
            return self._base_parent.base_ring()

        def base_parent(self):
            return self._base_parent

        def underlying_parent(self):
            """Return the underlying base-operad Sage parent in this arity."""

            return self._base_parent

        def from_base(self, element) -> "ShiftedOperad.Element":
            if element.parent() is self._base_parent:
                return self.sum_of_terms((basis, coeff) for basis, coeff in element)
            else:
                base_coerced = self._base_parent.sum_of_terms(
                    (basis, coeff) for basis, coeff in element
                )
                return self.sum_of_terms(
                    (basis, coeff) for basis, coeff in base_coerced
                )

        def degree_on_basis(self, basis_element) -> int:
            if isinstance(basis_element, ShiftedOperad.Element):
                support = basis_element.support()
                if not support:
                    return 0
                basis_element = next(iter(support))
            base_degree = self._base_parent.degree_on_basis(basis_element)
            return base_degree + self.factory.shift_degree * (self.arity() - 1)

        def compose(
            self, x: "ShiftedOperad.Element", i: int, y: "ShiftedOperad.Element"
        ) -> "ShiftedOperad.Element":
            return self.factory.compose(x, i, y)

        def unit(self) -> "ShiftedOperad.Element":
            return self.factory.unit(self.base_ring())

    class Element(CombinatorialFreeModule.Element):
        """Element wrapper carrying shifted operad structure maps."""

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "ShiftedOperad.Element":
            return self.parent().boundary(self)

        def permute(self, sigma) -> "ShiftedOperad.Element":
            if isinstance(sigma, (list, tuple)):
                sigma = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity()}. Got {sigma} ({type(sigma)})."
                )

            base_permuted = (
                self.parent()
                .base_parent()
                .sum_of_terms((basis, coeff) for basis, coeff in self)
                .permute(sigma)
            )
            sign = shifted_permutation_sign(self.parent().factory.shift_degree, sigma)
            return self.parent().from_base(sign * base_permuted)

        def base_element(self):
            """Return the underlying element in the base operad component."""

            return (
                self.parent()
                .base_parent()
                .sum_of_terms((basis, coeff) for basis, coeff in self)
            )

        def underlying_element(self):
            """Return the underlying element in the base operad component."""

            return self.base_element()


ShiftedOperad.Component.Element = ShiftedOperad.Element
