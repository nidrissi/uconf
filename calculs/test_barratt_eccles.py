"""Regression tests for the Barratt-Eccles operad."""

import math

import pytest

from uconf import BarrattEccles


def _degree_matches(element, d: int) -> bool:
    parent = element.parent()
    return all(parent.degree_on_basis(key) == d for key in element.support())


def _canonical_basis_key(basis):
    if not isinstance(basis, tuple):
        return basis

    canonical = []
    for item in basis:
        if hasattr(item, "tuple"):
            canonical.append(tuple(item.tuple()))
        else:
            canonical.append(item)
    return tuple(canonical)


def _as_dict(x):
    return {_canonical_basis_key(basis): coeff for basis, coeff in x}


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("d", range(0, 3))
def test_be_basis(r: int, d: int) -> None:
    basis = list(BarrattEccles(r).basis_it(d))
    if r == 1:
        expected_size = 1 if d == 0 else 0
    else:
        expected_size = math.factorial(r) * (math.factorial(r) - 1) ** d
    assert len(basis) == expected_size, f"Unexpected basis size for r={r}, d={d}."

    for el in basis:
        assert isinstance(
            el, BarrattEccles.Element
        ), "Non-BarrattEccles element found in basis_it"
        assert el != el.parent().zero(), "Zero element found in basis_it"
        assert el.arity() == r, "Element with incorrect arity in basis_it"
        assert _degree_matches(
            el, d
        ), "Element with incorrect degree in BarrattEccles basis_it"


def test_barratt_eccles_unit() -> None:
    u = BarrattEccles.unit()
    assert _as_dict(u) == {
        ((1,),): 1
    }, "Barratt-Eccles unit should be identity in arity 1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_barratt_eccles_permutation_action(sigma: list[int]) -> None:
    e2 = BarrattEccles(2)
    x = e2(((1, 2),))
    swapped = x.permute(sigma)
    assert _as_dict(swapped) == {
        ((1, 2),): 1
    }, "Unexpected transposed Barratt-Eccles term."
    assert _as_dict(swapped.permute(sigma)) == _as_dict(
        x
    ), "Applying the same transposition twice should return the original element."


def test_barratt_eccles_basic_composition() -> None:
    e2 = BarrattEccles(2)
    x = e2(((1, 2),))
    composed = BarrattEccles.compose(x, 1, x)
    assert _as_dict(composed) == {
        ((3, 2, 1),): 1
    }, "Expected canonical key of ((1,3),) in arity 3."


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_barratt_eccles_operadic_unit_axioms(input_pos: int) -> None:
    e3 = BarrattEccles(3)
    x = e3(((1, 2, 3),))
    one = BarrattEccles.unit()

    left = BarrattEccles.compose(one, 1, x)
    right = BarrattEccles.compose(x, input_pos, one)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed for Barratt-Eccles."
    assert _as_dict(right) == x_dict, f"Right unit axiom failed at input {input_pos}."