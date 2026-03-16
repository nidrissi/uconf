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
from uconf.algebraic.free_algebra import _module_basis_keys_in_degree
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
        left_operad_parent = self.left_algebra.operad_cls(
            arity, self.module.base_ring()
        )
        right_operad_parent = self.right_algebra.operad_cls(
            arity, self.module.base_ring()
        )

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
                        result += self.module.term(
                            (left_out_basis, right_out_basis)
                        ) * (scalar * left_out_coeff * right_out_coeff)

        return result

    def basis_it(self, d: int) -> Iterator:
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
        for d_left in range(d + 1):
            d_right = d - d_left
            left_keys = list(_module_basis_keys_in_degree(left_mod, d_left))
            if not left_keys:
                continue
            right_keys = list(_module_basis_keys_in_degree(right_mod, d_right))
            if not right_keys:
                continue
            for left_key in left_keys:
                for right_key in right_keys:
                    yield self.module.term((left_key, right_key))

    def boundary(self, a):
        """Tensor differential induced from the two dg-module differentials."""
        x = self.module(a)
        result = self.module.zero()

        for basis, coeff in x:
            left_basis, right_basis = basis
            left_term = self.left_module.term(left_basis)
            right_term = self.right_module.term(right_basis)

            left_degree = self.left_module.degree_on_basis(left_basis)
            sign = -1 if left_degree % 2 else 1

            left_boundary = self.left_algebra.boundary(left_term)
            right_boundary = self.right_algebra.boundary(right_term)

            for new_left_basis, left_coeff in left_boundary:
                result += self.module.term((new_left_basis, right_basis)) * (
                    coeff * left_coeff
                )
            for new_right_basis, right_coeff in right_boundary:
                result += self.module.term((left_basis, new_right_basis)) * (
                    coeff * sign * right_coeff
                )

        return result
