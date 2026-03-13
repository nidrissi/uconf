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

from typing import Any, Iterator

from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    QQ,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    UniqueRepresentation,
    tensor,
)

from uconf.core.operad import OperadLike
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.quasi_planar import QuasiPlanarMixin


class HadamardProduct(UniqueRepresentation):
    """Factory for Hadamard-product operad components."""

    def __init__(
        self,
        left_operad_cls: OperadLike,
        right_operad_cls: OperadLike,
    ):
        self.left_operad_cls = left_operad_cls
        self.right_operad_cls = right_operad_cls
        self.name = f"{left_operad_cls.name}⊙{right_operad_cls.name}"

    @property
    def connectivity(self) -> int:
        """Connectivity of the Hadamard product.

        ``(P ⊙ Q)(n)`` lives in degrees >= (k_P + k_Q)*(n-1) since the degree
        of a tensor ``(p, q)`` is ``deg_P(p) + deg_Q(q) >= k_P*(n-1) + k_Q*(n-1)``.
        """
        k_left = getattr(self.left_operad_cls, "connectivity", 0)
        k_right = getattr(self.right_operad_cls, "connectivity", 0)
        return k_left + k_right

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

    class Component(QuasiPlanarMixin, CombinatorialFreeModule):
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
            # OperadProtocol expects a `name` attribute on each component.
            self.name = factory.name
            self._arity = int(n)
            self._left_parent = factory.left_operad_cls(n, base_ring)
            self._right_parent = factory.right_operad_cls(n, base_ring)
            self._symmetric_group = SymmetricGroup(self._arity)
            self._symmetric_group_algebra = SymmetricGroupAlgebra(base_ring, n)
            self.rename(name)
            self.boundary = self.module_morphism(
                on_basis=self._boundary_on_basis,
                codomain=self,
            )
            # Set up planarize if the right factor supports it
            if hasattr(self._right_parent, "planarize"):
                self.planarize = self.module_morphism(
                    on_basis=self._planarize_on_basis,
                    codomain=tensor([self, self._symmetric_group_algebra]),
                )

        def _planarize_on_basis(self, basis_element: tuple) -> Any:
            """Planarize via the right factor's quasi-planar structure.

            For a basis element ``(p, q)`` where ``q`` is in an operad with
            ``planarize``, we decompose ``q = q_pl · σ`` and return
            ``(p · σ⁻¹, q_pl) ⊗ σ``.
            """
            left_basis, right_basis = basis_element
            right_parent = self._right_parent
            left_parent = self._left_parent

            # Planarize the right factor: q -> q_pl ⊗ σ
            right_elem = right_parent.term(right_basis)
            right_planarized = right_parent.planarize(right_elem)  # type: ignore[attr-defined]

            target = tensor([self, self._symmetric_group_algebra])
            result = target.zero()

            for tensor_basis, coeff in right_planarized:
                right_pl_key, sigma_key = tensor_basis
                # sigma_key is a permutation in S_n
                sigma = self._symmetric_group(sigma_key)
                sigma_inv = sigma.inverse()

                # Permute the left factor by σ⁻¹
                left_elem = left_parent.term(left_basis)
                left_permuted = left_elem.permute(sigma_inv)

                for new_left_key, left_coeff in left_permuted:
                    had_key = (new_left_key, right_pl_key)
                    result += (coeff * left_coeff) * self.term(had_key).tensor(
                        self._symmetric_group_algebra.term(sigma_key)
                    )

            return result

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

        @property
        def connectivity(self) -> int:
            """Connectivity of this Hadamard component (sum of left and right)."""
            return self.factory.connectivity

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

        def planar_basis_it(self, d: int) -> Iterator["HadamardProduct.Element"]:
            """Iterate over planar basis elements of degree ``d``.

            A pair ``(left_key, right_key)`` is *planar* when ``right_key``
            is a planar element of the right factor.  Requires the right
            factor to implement ``planar_basis_it``.  The left factor is
            iterated with ``basis_it(d_left)`` (degree-indexed) when
            available, or with ``basis_it()`` filtered by degree otherwise.
            """
            if not hasattr(self._right_parent, "planar_basis_it"):
                return

            left_parent = self._left_parent
            right_parent = self._right_parent

            for d_right in range(d + 1):
                d_left = d - d_right
                right_elems = list(right_parent.planar_basis_it(d_right))  # type: ignore[attr-defined]
                if not right_elems:
                    continue

                # Gather left elements at degree d_left.
                left_elems = []
                if hasattr(left_parent, "basis_it"):
                    try:
                        left_elems = list(left_parent.basis_it(d_left))  # type: ignore[attr-defined]
                    except TypeError:
                        # Degree-free basis_it (e.g. Lie)
                        for elem in left_parent.basis_it():  # type: ignore[attr-defined]
                            for key in elem.support():
                                if left_parent.degree_on_basis(key) == d_left:
                                    left_elems.append(left_parent.term(key))
                else:
                    try:
                        for key in left_parent.basis():  # type: ignore[attr-defined]
                            if left_parent.degree_on_basis(key) == d_left:
                                left_elems.append(left_parent.term(key))
                    except (AttributeError, NotImplementedError):
                        pass

                for left_elem in left_elems:
                    for right_elem in right_elems:
                        yield self.from_factors(left_elem, right_elem)

        def from_factors(
            self, left_element, right_element
        ) -> "HadamardProduct.Element":
            left_parent = left_element.parent()
            right_parent = right_element.parent()

            if left_parent.arity() != self.arity():
                raise TypeError(
                    f"Left element arity {left_parent.arity()} does not match target arity {self.arity()}."
                )
            if right_parent.arity() != self.arity():
                raise TypeError(
                    f"Right element arity {right_parent.arity()} does not match target arity {self.arity()}."
                )
            if left_parent.base_ring() != self.base_ring():
                raise TypeError("Left element must have the same base ring as target.")
            if right_parent.base_ring() != self.base_ring():
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

    class Element(
        ParentedElementMixin["HadamardProduct.Component"],
        CombinatorialFreeModule.Element,
    ):
        """Element wrapper carrying Hadamard-operad structure maps."""

        def arity(self) -> int:
            return self._parent().arity()

        def boundary(self) -> "HadamardProduct.Element":
            parent = self._parent()
            return parent.boundary(self)

        def permute(self, sigma) -> "HadamardProduct.Element":
            parent = self._parent()
            if isinstance(sigma, (list, tuple)):
                sigma = parent._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{parent.arity()}. Got {sigma} ({type(sigma)})."
                )

            result = parent.zero()
            for basis, coeff in self:
                left_basis, right_basis = basis
                left_term = parent.left_parent().term(left_basis).permute(sigma)
                right_term = parent.right_parent().term(right_basis).permute(sigma)
                result += parent.from_factors(left_term, right_term) * coeff
            return result


HadamardProduct.Component.Element = HadamardProduct.Element
