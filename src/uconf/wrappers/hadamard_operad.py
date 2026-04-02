"""Hadamard product wrapper for operads.

For operads ``P`` and ``Q``, this module implements the aritywise tensor-product
operad ``P ⊙ Q``:

- ``(P ⊙ Q)(n) = P(n) ⊗ Q(n)`` as graded modules,
- differential ``d(p ⊗ q) = dp ⊗ q + (-1)^|p| p ⊗ dq``,
- symmetric action diagonal: ``(p ⊗ q)·σ = (p·σ) ⊗ (q·σ)``,
- partial composition diagonal:
  ``(p ⊗ q) ∘_i (p' ⊗ q') = (-1)^{|q|·|p'|} (p ∘_i p') ⊗ (q ∘_i q')``.
"""

from __future__ import annotations

from typing import Any, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    UniqueRepresentation,
    cached_method,
    tensor,
)

from uconf.core.operad import OperadLike
from uconf.core.display import latex_linear_combination
from uconf.core.parented_element import ParentedElementMixin
from uconf.core.quasi_planar import QuasiPlanarMixin
from uconf.core.signs import sign_from_exponent


def _component_basis_in_degree(component, d: int) -> list:
    """Return a list of degree-*d* basis elements from an operad/cooperad component.

    Tries ``component.basis_iter(d)`` first; falls back to ``basis()`` filtered by
    ``degree_on_basis``.
    """
    basis_it_fn = getattr(component, "basis_iter", None)
    if basis_it_fn is not None:
        return list(basis_it_fn(d))
    # Fallback: iterate basis() and filter by degree
    result = []
    for key in component.basis():
        if component.degree_on_basis(key) == d:
            result.append(component.term(key))
    return result


def _min_component_degree(component, arity: int) -> int:
    """Return the minimum possible degree in a fixed-arity component.

    If the underlying operad has connectivity ``k`` (i.e. degrees in arity
    ``n`` are bounded below by ``k*(n-1)``), then this component has minimum
    degree ``k*(arity-1)``.
    """
    connectivity = int(getattr(component, "connectivity", 0))
    return connectivity * (arity - 1)


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

    def _repr_(self) -> str:
        return self.name

    def _repr_latex_(self) -> str:
        left = getattr(self.left_operad_cls, "name", "P")
        right = getattr(self.right_operad_cls, "name", "Q")
        return f"{left} \\odot {right}"

    @property
    def connectivity(self) -> int:
        """Connectivity of the Hadamard product.

        ``(P ⊙ Q)(n)`` lives in degrees >= (k_P + k_Q)*(n-1) since the degree
        of a tensor ``(p, q)`` is ``deg_P(p) + deg_Q(q) >= k_P*(n-1) + k_Q*(n-1)``.
        """
        k_left = getattr(self.left_operad_cls, "connectivity", 0)
        k_right = getattr(self.right_operad_cls, "connectivity", 0)
        return k_left + k_right

    def __call__(self, n: int, base_ring) -> "HadamardProduct.Component":
        return HadamardProduct.Component(self, n, base_ring)

    def unit(self, base_ring) -> "HadamardProduct.Element":
        component = self(1, base_ring)
        return component.from_factors(
            self.left_operad_cls.unit(base_ring),
            self.right_operad_cls.unit(base_ring),
        )

    def unit_key(self) -> tuple:
        """Return the basis key of the unit element in arity ``1``."""
        return (self.left_operad_cls.unit_key(), self.right_operad_cls.unit_key())

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
            right_x_degree = right_x.degree_on_basis(right_x_basis)
            for y_basis, y_coeff in y:
                left_y_basis, right_y_basis = y_basis
                left_y_term = left_y.term(left_y_basis)
                right_y_term = right_y.term(right_y_basis)
                left_y_degree = left_y.degree_on_basis(left_y_basis)

                # Koszul sign from commuting right_x past left_y:
                # (a⊗b) ∘_i (c⊗d) = (-1)^{|b|·|c|} (a∘_i c) ⊗ (b∘_i d)
                koszul_sign = sign_from_exponent(right_x_degree * left_y_degree)

                left_composed = self.left_operad_cls.compose(left_x_term, i, left_y_term)
                right_composed = self.right_operad_cls.compose(right_x_term, i, right_y_term)

                for left_basis, left_coeff in left_composed:
                    for right_basis, right_coeff in right_composed:
                        accumulated += (
                            koszul_sign * x_coeff * y_coeff * left_coeff * right_coeff
                        ) * target.term((left_basis, right_basis))

        return accumulated

    class Component(QuasiPlanarMixin, CombinatorialFreeModule):
        """A fixed-arity component of ``P ⊙ Q``."""

        def __init__(self, factory: "HadamardProduct", n: int, base_ring):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            name = f"{factory.name}{n}"
            super().__init__(
                base_ring,
                tuple,
                prefix=name,
                category=GradedModulesWithBasis(base_ring),
            )
            self.factory = factory
            # OperadComponent expects a `name` attribute on each component.
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

        @cached_method
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
            right_planarized = right_parent.planarize(right_elem)

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

            left_validate = getattr(self._left_parent, "_validate_basis_key", None)
            right_validate = getattr(self._right_parent, "_validate_basis_key", None)
            if left_validate is not None:
                left_basis = left_validate(left_basis)
            if right_validate is not None:
                right_basis = right_validate(right_basis)

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

            raise TypeError(
                f"Expected dict, tuple/list, or HadamardProduct.Element; got {type(x).__name__}: {x!r}."
            )

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
                result += self.term((left_basis, new_right_basis)) * (sign * right_coeff)

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

        def _repr_term(self, basis_element: tuple) -> str:
            left_basis, right_basis = basis_element
            left_term = getattr(self._left_parent, "_repr_term", None)
            right_term = getattr(self._right_parent, "_repr_term", None)

            left_str = left_term(left_basis) if callable(left_term) else str(left_basis)
            right_str = right_term(right_basis) if callable(right_term) else str(right_basis)
            return f"{left_str} ⊙ {right_str}"

        def _latex_term(self, basis_element: tuple) -> str:
            left_basis, right_basis = basis_element
            left_term = getattr(self._left_parent, "_latex_term", None)
            right_term = getattr(self._right_parent, "_latex_term", None)

            left_ltx = left_term(left_basis) if callable(left_term) else str(left_basis)
            right_ltx = right_term(right_basis) if callable(right_term) else str(right_basis)
            return f"{left_ltx} \\odot {right_ltx}"

        def basis_iter(self, d: int) -> Iterator["HadamardProduct.Element"]:
            """Iterate over all basis elements of degree ``d``.

            Yields all pairs ``(left_key, right_key)`` with
            ``deg_P(left_key) + deg_Q(right_key) = d``.  Both factors are
            enumerated via ``basis_iter(d_left)`` (if available) or by
            filtering ``basis()``.

            Args:
                d: Total degree to enumerate.

            Yields:
                Elements of this Hadamard component with degree ``d``.
            """
            left_parent = self._left_parent
            right_parent = self._right_parent

            min_d_left = _min_component_degree(left_parent, self._arity)
            min_d_right = _min_component_degree(right_parent, self._arity)
            max_d_left = d - min_d_right
            if max_d_left < min_d_left:
                return

            for d_left in range(min_d_left, max_d_left + 1):
                d_right = d - d_left
                left_elems = list(_component_basis_in_degree(left_parent, d_left))
                if not left_elems:
                    continue
                right_elems = list(_component_basis_in_degree(right_parent, d_right))
                if not right_elems:
                    continue
                for left_elem in left_elems:
                    for right_elem in right_elems:
                        yield self.from_factors(left_elem, right_elem)

        def planar_basis_iter(self, d: int) -> Iterator["HadamardProduct.Element"]:
            """Iterate over planar basis elements of degree ``d``.

            A pair ``(left_key, right_key)`` is *planar* when ``right_key``
            is a planar element of the right factor.  Requires the right
            factor to implement ``planar_basis_iter``.  The left factor is
            iterated with ``basis_iter(d_left)`` (degree-indexed).
            """
            left_parent = self._left_parent
            right_parent = self._right_parent

            if not hasattr(right_parent, "planar_basis_iter"):
                raise NotImplementedError("Right parent does not support planar_basis_iter.")

            min_d_left = _min_component_degree(left_parent, self._arity)
            min_d_right = _min_component_degree(right_parent, self._arity)
            max_d_right = d - min_d_left
            if max_d_right < min_d_right:
                return

            for d_right in range(min_d_right, max_d_right + 1):
                d_left = d - d_right
                right_elems = list(right_parent.planar_basis_iter(d_right))
                if not right_elems:
                    continue

                left_elems = list(_component_basis_in_degree(left_parent, d_left))

                for left_elem in left_elems:
                    for right_elem in right_elems:
                        yield self.from_factors(left_elem, right_elem)

        @cached_method
        def graded_basis(self, d: int) -> Family:
            """Return the ``Family`` of all basis elements in degree ``d``."""
            return Family(self.basis_iter(d))

        @cached_method
        def graded_planar_basis(self, d: int) -> Family:
            """Return the ``Family`` of planar basis elements in degree ``d``."""
            return Family(self.planar_basis_iter(d))

        def from_factors(self, left_element, right_element) -> "HadamardProduct.Element":
            left_parent = left_element.parent()
            right_parent = right_element.parent()

            if left_parent.arity() != self.arity():
                raise TypeError(
                    f"Left element arity {left_parent.arity()} does not match {self.arity()}."
                )
            if right_parent.arity() != self.arity():
                raise TypeError(
                    f"Right element arity {right_parent.arity()} does not match {self.arity()}."
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

        def unit_key(self) -> tuple:
            """Return the basis key of the unit element in arity ``1``."""
            return self.factory.unit_key()

    class Element(
        ParentedElementMixin["HadamardProduct.Component"],
        CombinatorialFreeModule.Element,
    ):
        """Element wrapper carrying Hadamard-operad structure maps."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "HadamardProduct.Element":
            parent = self.parent()
            return parent.boundary(self)

        def planarize(self):
            """Project to planar representative tensored with a group element."""
            parent = self.parent()
            return parent.planarize(self)

        def permute(self, sigma) -> "HadamardProduct.Element":
            parent = self.parent()
            if isinstance(sigma, (list, tuple)):
                sigma = parent._symmetric_group(sigma)
            elif not (hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group):
                raise TypeError(
                    f"Permutation must be a list, tuple, or S_{parent.arity()} element; "
                    f"got {type(sigma).__name__}: {sigma!r}."
                )

            result = parent.zero()
            for basis, coeff in self:
                left_basis, right_basis = basis
                left_term = parent.left_parent().term(left_basis).permute(sigma)
                right_term = parent.right_parent().term(right_basis).permute(sigma)
                result += parent.from_factors(left_term, right_term) * coeff
            return result


HadamardProduct.Component.Element = HadamardProduct.Element
