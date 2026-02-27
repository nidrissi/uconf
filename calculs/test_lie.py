"""Regression tests for the Lie operad implementation.

Permutation inputs are passed as one-line notation lists, e.g. ``[2, 3, 1]``.
Using tuples may be interpreted by Sage in cycle-style constructors and can lead
to ambiguous behavior in tests.
"""

import pytest

from uconf import Lie


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


def test_unit() -> None:
    u = Lie.unit()
    assert _as_dict(u) == {(): 1}, "Unit should be the arity-1 generator x1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_antisymmetry_via_permutation(sigma: list[int]) -> None:
    l2 = Lie(2)
    bracket = l2((1,))
    assert (
        bracket.permute(sigma) == -bracket
    ), "Swapping x1 and x2 must negate [x1,x2]."


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
    assert _as_dict(right) == x_dict, (
        f"Right unit axiom failed at input {input_pos}: x∘{input_pos} 1 = x."
    )


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
