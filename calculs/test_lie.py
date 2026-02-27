"""Regression tests for the Lie operad implementation.

Permutation inputs are passed as one-line notation lists, e.g. ``[2, 3, 1]``.
Using tuples may be interpreted by Sage in cycle-style constructors and can lead
to ambiguous behavior in tests.
"""

import math

import pytest

from uconf import Lie


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


@pytest.mark.parametrize("n", range(0, 6))
def test_basis_it_size(n: int) -> None:
    basis = list(Lie(n).basis_it())
    if n == 0:
        expected_size = 0
    else:
        expected_size = math.factorial(n - 1)
    assert len(basis) == expected_size, "Unexpected basis size in arity n."


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_basis_in_arity_matches_iterator(n: int) -> None:
    from_iterator = list(Lie(n).basis_it())
    from_static = list(Lie.basis_in_arity(n))
    assert len(from_iterator) == len(from_static)
    assert [_as_dict(x) for x in from_iterator] == [_as_dict(x) for x in from_static]


def test_unit() -> None:
    u = Lie.unit()
    assert _as_dict(u) == {(): 1}, "Unit should be the arity-1 generator x1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_antisymmetry_via_permutation(sigma: list[int]) -> None:
    l2 = Lie(2)
    bracket = l2((1,))
    assert bracket.permute(sigma) == -bracket, "Swapping x1 and x2 must negate [x1,x2]."


def test_basic_composition() -> None:
    l2 = Lie(2)
    bracket = l2((1,))
    composed = Lie.compose(bracket, 1, bracket)
    target = composed.parent()
    expected = target((1, 2)) - target((2, 1))
    assert _as_dict(composed) == _as_dict(
        expected
    ), "[x1,[x2,x3]] - [x2,[x1,x3]] expected."


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_operadic_unit_axioms(input_pos: int) -> None:
    l3 = Lie(3)
    x = l3((1, 2))
    unit = Lie.unit()

    left = Lie.compose(unit, 1, x)
    right = Lie.compose(x, input_pos, unit)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed: 1∘1 x = x."
    assert (
        _as_dict(right) == x_dict
    ), f"Right unit axiom failed at input {input_pos}: x∘{input_pos} 1 = x."


def test_jacobi_identity() -> None:
    l2 = Lie(2)
    bracket = l2((1,))

    jacobi_generator = Lie.compose(bracket, 1, bracket)
    jacobi = (
        jacobi_generator
        + jacobi_generator.permute([2, 3, 1])
        + jacobi_generator.permute([3, 1, 2])
    )
    assert _as_dict(jacobi) == {}, "Jacobi identity failed in arity 4."
