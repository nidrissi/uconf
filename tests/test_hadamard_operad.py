"""Tests for the Hadamard-product operad wrapper."""

import pytest
from sage.all import QQ, GF

from uconf import HadamardProduct, Lie, ShiftedOperad, Surjection


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


def test_degree_is_sum_of_factor_degrees() -> None:
    had = HadamardProduct(Lie, Surjection)
    h2 = had(2, QQ)
    basis = ((1,), (1, 2, 1))
    assert h2.degree_on_basis(basis) == 1


def test_boundary_uses_tensor_sign_rule() -> None:
    had = HadamardProduct(Surjection, Surjection)
    h2 = had(2, QQ)

    left_basis = (1, 2, 1)
    right_basis = (1, 2, 1)
    x = h2((left_basis, right_basis))

    left_parent = Surjection(2, QQ)
    right_parent = Surjection(2, QQ)

    expected = h2.zero()
    left_boundary = left_parent.boundary(left_parent(left_basis))
    right_boundary = right_parent.boundary(right_parent(right_basis))

    for new_left_basis, coeff in left_boundary:
        expected += h2((new_left_basis, right_basis)) * coeff

    for new_right_basis, coeff in right_boundary:
        expected += h2((left_basis, new_right_basis)) * (-coeff)

    assert _as_dict(x.boundary()) == _as_dict(expected)


def test_compose_is_diagonal_on_factors() -> None:
    had = HadamardProduct(Surjection, Surjection)
    h2 = had(2, QQ)

    x = h2(((1, 2), (1, 2)))
    y = h2(((1, 2, 1), (1, 2)))

    composed = had.compose(x, 1, y)

    left_composed = Surjection.compose(Surjection(2, QQ)((1, 2)), 1, Surjection(2, QQ)((1, 2, 1)))
    right_composed = Surjection.compose(Surjection(2, QQ)((1, 2)), 1, Surjection(2, QQ)((1, 2)))

    expected = had(3, QQ).from_factors(left_composed, right_composed)
    assert _as_dict(composed) == _as_dict(expected)


def test_permutation_is_diagonal_action() -> None:
    had = HadamardProduct(Lie, Lie)
    h2 = had(2, QQ)

    x = h2(((1,), (1,)))
    sigma = [2, 1]

    left = Lie(2, QQ)((1,)).permute(sigma)
    right = Lie(2, QQ)((1,)).permute(sigma)
    expected = h2.from_factors(left, right)

    assert _as_dict(x.permute(sigma)) == _as_dict(expected)


def test_compose_requires_same_base_ring() -> None:
    had = HadamardProduct(Surjection, Surjection)
    x = had(2, GF(2))(((1, 2), (1, 2)))
    y = had(2, QQ)(((1, 2), (1, 2)))

    with pytest.raises(TypeError, match="same base ring"):
        had.compose(x, 1, y)


def test_hadamard_accepts_shifted_operad_provider() -> None:
    shifted_lie = ShiftedOperad(Lie, -1)
    had = HadamardProduct(Surjection, shifted_lie)

    x = had(2, QQ)(((1, 2), (1,)))
    y = had(2, QQ)(((1, 2, 1), (1,)))
    composed = had.compose(x, 1, y)

    assert x.arity() == 2
    assert composed.arity() == 3


def test_element_repr_latex_exists() -> None:
    had = HadamardProduct(Lie, Surjection)
    x = had(2, QQ)(((1,), (1, 2, 1)))
    ltx = x._repr_latex_()
    assert ltx
