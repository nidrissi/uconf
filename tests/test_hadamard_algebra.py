"""Tests for the Hadamard tensor algebra wrapper."""

import pytest
from sage.all import QQ, ZZ, tensor
from typing import cast

from uconf import (
    Associative,
    Commutative,
    HadamardProduct,
    HadamardTensorAlgebra,
    OperadAlgebra,
)
from uconf.models.simplicial import SimplicialChains


def _as_dict(x):
    return {basis: coeff for basis, coeff in x if coeff != 0}


class TrivialAssAlgebra(OperadAlgebra):
    """Associative algebra structure on the 1-dimensional module k."""

    def __init__(self, base_ring=QQ):
        module = Commutative(1, base_ring=base_ring)
        super().__init__(module, Associative, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        result = self.module.zero()
        for _p_basis, p_coeff in p_element:
            coeff = p_coeff
            for a_elem in algebra_elements:
                for _a_basis, a_coeff in a_elem:
                    coeff *= a_coeff
            result += coeff * self.module(())
        return result


class UnaryChainAlgebra(OperadAlgebra):
    """A minimal algebra wrapper used to test tensor differentials."""

    def __init__(self):
        module = SimplicialChains(QQ)
        super().__init__(module, Associative, self._structure_map)

    def _structure_map(self, p_element, algebra_elements):
        if p_element.arity() != 1:
            return self.module.zero()
        out = self.module.zero()
        for _basis, coeff in p_element:
            out += coeff * algebra_elements[0]
        return out


def test_hadamard_tensor_algebra_construction() -> None:
    left = TrivialAssAlgebra()
    right = TrivialAssAlgebra()

    had_alg = HadamardTensorAlgebra(left, right)
    had = cast(HadamardProduct, had_alg.operad_cls)

    assert had.left_operad_cls is Associative
    assert had.right_operad_cls is Associative


def test_unit_action_on_tensor_module() -> None:
    alg = TrivialAssAlgebra()
    had_alg = HadamardTensorAlgebra(alg, alg)

    x = tensor((alg(()), alg(())))
    unit = had_alg.operad_cls.unit(QQ)

    assert had_alg.act(unit, [x]) == x


def test_binary_action_multiplies_tensor_scalars() -> None:
    alg = TrivialAssAlgebra()
    had_alg = HadamardTensorAlgebra(alg, alg)
    had = cast(HadamardProduct, had_alg.operad_cls)

    x = tensor((alg(()), alg(())))
    p = had(2, QQ)(((1, 2), (1, 2)))

    result = had_alg.act(p, [2 * x, 3 * x])
    assert result == 6 * x


def test_boundary_uses_tensor_sign_rule() -> None:
    alg = UnaryChainAlgebra()
    had_alg = HadamardTensorAlgebra(alg, alg)
    x = tensor((alg((0, 1)), alg((0, 1))))

    expected = had_alg.module.zero()
    expected += tensor((alg((1,)), alg((0, 1))))
    expected -= tensor((alg((0,)), alg((0, 1))))
    expected -= tensor((alg((0, 1)), alg((1,))))
    expected += tensor((alg((0, 1)), alg((0,))))

    assert _as_dict(had_alg.boundary(x)) == _as_dict(expected)


def test_constructor_requires_same_base_ring() -> None:
    left = TrivialAssAlgebra(base_ring=QQ)
    right = TrivialAssAlgebra(base_ring=ZZ)

    with pytest.raises(TypeError, match="same base ring"):
        HadamardTensorAlgebra(left, right)
