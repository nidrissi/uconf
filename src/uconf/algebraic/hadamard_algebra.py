"""Hadamard-product algebra wrapper.

Given a ``P``-algebra ``A`` and a ``Q``-algebra ``B``, this module builds the
canonical ``(P ⊙ Q)``-algebra structure on ``A ⊗ B`` where ``⊙`` denotes the
Hadamard product of operads.

For homogeneous pure tensors, the action is

    (p ⊗ q) · ((a_1 ⊗ b_1), ..., (a_n ⊗ b_n))
      = (p·(a_1,...,a_n)) ⊗ (q·(b_1,...,b_n)),

extended multilinearly.  The differential on ``A ⊗ B`` is the tensor one:

    d(a ⊗ b) = da ⊗ b + (-1)^|a| a ⊗ db.
"""

from __future__ import annotations

import itertools
from typing import Iterator

from sage.all import tensor

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.tree_module import (
    _inner_weight_on_key,
    _module_basis_keys_in_degree,
    _module_basis_keys_in_weight_and_degree,
)
from uconf.wrappers.hadamard_operad import HadamardProduct


class HadamardTensorAlgebra(OperadAlgebra):
    """The canonical ``(P ⊙ Q)``-algebra on ``A ⊗ B``.

    Args:
        left_algebra: A ``P``-algebra wrapper.
        right_algebra: A ``Q``-algebra wrapper.

    """

    def __init__(self, left_algebra: OperadAlgebra, right_algebra: OperadAlgebra):
        left_ring = left_algebra.module.base_ring()
        right_ring = right_algebra.module.base_ring()
        if left_ring != right_ring:
            raise TypeError("Both algebra modules must have the same base ring.")

        hadamard_operad = HadamardProduct(
            left_algebra.operad_cls,
            right_algebra.operad_cls,
        )

        self.left_algebra = left_algebra
        self.right_algebra = right_algebra
        self.left_module = left_algebra.module
        self.right_module = right_algebra.module

        super().__init__(
            module=tensor([self.left_module, self.right_module]),
            operad_cls=hadamard_operad,
            structure_map=self._act_impl,
        )

        self.module.basis_iter = self.basis_iter
        self.module.boundary = self._tensor_boundary_morphism()
        # Connectivity is additive for tensor products: deg(a⊗b) = deg(a) + deg(b),
        # so min(deg(a⊗b)) = min(deg(a)) + min(deg(b)).
        self.module.connectivity = int(getattr(self.left_module, "connectivity", 0)) + int(
            getattr(self.right_module, "connectivity", 0)
        )
        self.module.degree_on_basis = lambda key: (
            self.left_module.degree_on_basis(key[0]) + self.right_module.degree_on_basis(key[1])
        )
        # Weight API: additive over tensor factors.
        self.module.basis_weight_iter = self.basis_weight_iter
        self.module.graded_basis_by_weight = self.graded_basis_by_weight
        self.module._weight_on_basis = lambda key: (
            _inner_weight_on_key(self.left_module, key[0])
            + _inner_weight_on_key(self.right_module, key[1])
        )

    def _act_impl(self, p_element, algebra_elements):
        if p_element.parent().factory is not self.operad_cls:
            raise TypeError("p_element must belong to this Hadamard operad.")

        if p_element.parent().base_ring() != self.module.base_ring():
            raise TypeError("Operad element and algebra module must share base ring.")

        tensor_args = [self.module(a) for a in algebra_elements]
        for a in tensor_args:
            if a.parent() != self.module:
                raise TypeError("All algebra elements must lie in the tensor module.")

        arity = p_element.arity()
        left_operad_parent = self.left_algebra.operad_cls(arity, self.module.base_ring())
        right_operad_parent = self.right_algebra.operad_cls(arity, self.module.base_ring())

        result = self.module.zero()
        arg_expansions = [list(arg) for arg in tensor_args]

        for had_basis, had_coeff in p_element:
            left_basis, right_basis = had_basis
            left_op = left_operad_parent.term(left_basis)
            right_op = right_operad_parent.term(right_basis)

            for selected_terms in itertools.product(*arg_expansions):
                scalar = had_coeff
                left_inputs = []
                right_inputs = []

                for tensor_basis, tensor_coeff in selected_terms:
                    left_key, right_key = tensor_basis
                    left_inputs.append(self.left_module.term(left_key))
                    right_inputs.append(self.right_module.term(right_key))
                    scalar *= tensor_coeff

                if scalar == 0:
                    continue

                left_value = self.left_algebra.act(left_op, left_inputs)
                right_value = self.right_algebra.act(right_op, right_inputs)

                for left_out_basis, left_out_coeff in left_value:
                    for right_out_basis, right_out_coeff in right_value:
                        result += self.module.term((left_out_basis, right_out_basis)) * (
                            scalar * left_out_coeff * right_out_coeff
                        )

        return result

    def basis_iter(self, d: int) -> Iterator:
        """Iterate over basis elements of degree *d*.

        Yields all ``(left_key, right_key)`` tensor-module basis elements with
        ``deg_A(left_key) + deg_B(right_key) = d``.  Both factors are
        enumerated via :func:`_module_basis_keys_in_degree`.

        Args:
            d: Homological degree to enumerate.

        Yields:
            Elements of the tensor module ``A ⊗ B`` with degree ``d``.
        """
        left_mod = self.left_module
        right_mod = self.right_module

        # TODO right now, most modules don't have a notion of connectivity, so we start from 0!
        min_d_left = left_mod.connectivity
        min_d_right = right_mod.connectivity
        max_d_left = d - min_d_right
        if max_d_left < min_d_left:
            return

        for d_left in range(min_d_left, max_d_left + 1):
            d_right = d - d_left
            left_keys = list(_module_basis_keys_in_degree(left_mod, d_left))
            if not left_keys:
                continue
            right_keys = list(_module_basis_keys_in_degree(right_mod, d_right))
            if not right_keys:
                continue
            for left_key in left_keys:
                for right_key in right_keys:
                    yield tensor((self.left_module(left_key), self.right_module(right_key)))

    def basis_weight_iter(self, d: int, w: int) -> Iterator:
        """Iterate over basis elements of degree ``d`` and weight ``w``.

        Yields all ``(left_key, right_key)`` tensor-module basis elements
        where ``deg(left) + deg(right) = d`` and
        ``weight(left) + weight(right) = w``.  Weights are obtained via
        :func:`_module_basis_keys_in_weight_and_degree` on each factor.

        Args:
            d: Homological degree.
            w: Weight to enumerate.

        Yields:
            Elements of the tensor module ``A ⊗ B`` with degree ``d`` and
            weight ``w``.
        """
        left_mod = self.left_module
        right_mod = self.right_module

        min_d_left = left_mod.connectivity
        min_d_right = right_mod.connectivity

        for w_left in range(0, w + 1):
            w_right = w - w_left
            max_d_left = d - min_d_right
            if max_d_left < min_d_left:
                continue
            for d_left in range(min_d_left, max_d_left + 1):
                d_right = d - d_left
                left_keys = list(
                    _module_basis_keys_in_weight_and_degree(left_mod, d_left, w_left)
                )
                if not left_keys:
                    continue
                right_keys = list(
                    _module_basis_keys_in_weight_and_degree(right_mod, d_right, w_right)
                )
                if not right_keys:
                    continue
                for left_key in left_keys:
                    for right_key in right_keys:
                        yield tensor(
                            (self.left_module(left_key), self.right_module(right_key))
                        )

    def graded_basis_by_weight(self, d: int, w: int):
        """Family of basis elements of degree ``d`` and weight ``w``."""
        from sage.all import Family

        return Family(self.basis_weight_iter(d, w))

    def _tensor_boundary_morphism(self):
        """Build a module morphism for the tensor differential on the module.

        This allows the tensor module's ``boundary`` to work as a proper Sage
        module morphism when called by external code (e.g.
        :class:`~uconf.algebraic.tree_module.TreeModule`).
        """
        return self.module.module_morphism(
            on_basis=self._boundary_on_tensor_basis, codomain=self.module
        )

    def _boundary_on_tensor_basis(self, basis_key):
        """Tensor differential on a single basis key ``(left_key, right_key)``."""
        left_basis, right_basis = basis_key

        left_degree = self.left_module.degree_on_basis(left_basis)
        sign = -1 if left_degree % 2 else 1

        left_boundary = self.left_algebra.boundary(self.left_module.term(left_basis))
        right_boundary = self.right_algebra.boundary(self.right_module.term(right_basis))

        result = self.module.zero()
        for new_left_basis, left_coeff in left_boundary:
            result += left_coeff * self.module.term((new_left_basis, right_basis))
        for new_right_basis, right_coeff in right_boundary:
            result += sign * right_coeff * self.module.term((left_basis, new_right_basis))
        return result

    def boundary(self, a):
        """Tensor differential induced from the two dg-module differentials."""
        x = self.module(a)
        result = self.module.zero()

        for basis, coeff in x:
            left_basis, right_basis = basis
            left_term = self.left_module(left_basis)
            right_term = self.right_module(right_basis)

            left_degree = self.left_module.degree_on_basis(left_basis)
            sign = -1 if left_degree % 2 else 1

            left_boundary = self.left_algebra.boundary(left_term)
            right_boundary = self.right_algebra.boundary(right_term)

            for new_left_basis, left_coeff in left_boundary:
                result += self.module.term((new_left_basis, right_basis)) * (coeff * left_coeff)
            for new_right_basis, right_coeff in right_boundary:
                result += self.module.term((left_basis, new_right_basis)) * (
                    coeff * sign * right_coeff
                )

        return result
