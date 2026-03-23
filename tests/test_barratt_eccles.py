"""Regression tests for the Barratt-Eccles operad."""

import math

import pytest
from sage.all import ZZ, QQ

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
    basis = list(BarrattEccles(r, QQ).basis_iter(d))
    if r == 1:
        expected_size = 1 if d == 0 else 0
    else:
        expected_size = math.factorial(r) * (math.factorial(r) - 1) ** d
    assert len(basis) == expected_size, f"Unexpected basis size for r={r}, d={d}."

    for el in basis:
        assert isinstance(el, BarrattEccles.Element), (
            "Non-BarrattEccles element found in basis_iter"
        )
        assert el != el.parent().zero(), "Zero element found in basis_iter"
        assert el.arity() == r, "Element with incorrect arity in basis_iter"
        assert _degree_matches(el, d), "Element with incorrect degree in BarrattEccles basis_iter"


def test_barratt_eccles_unit() -> None:
    u = BarrattEccles.unit(QQ)
    assert _as_dict(u) == {((1,),): 1}, "Barratt-Eccles unit should be identity in arity 1."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_barratt_eccles_permutation_action(sigma: list[int]) -> None:
    e2 = BarrattEccles(2, QQ)
    x = e2(((1, 2),))
    swapped = x.permute(sigma)
    assert _as_dict(swapped) == {((1, 2),): 1}, "Unexpected transposed Barratt-Eccles term."
    assert _as_dict(swapped.permute(sigma)) == _as_dict(x), (
        "Applying the same transposition twice should return the original element."
    )


def test_barratt_eccles_basic_composition() -> None:
    e2 = BarrattEccles(2, QQ)
    x = e2(((1, 2),))
    composed = BarrattEccles.compose(x, 1, x)
    assert _as_dict(composed) == {((3, 2, 1),): 1}, "Expected canonical key of ((1,3),) in arity 3."


def test_barratt_eccles_compose_requires_same_base_ring() -> None:
    x = BarrattEccles(2, QQ)(((1, 2),))
    y = BarrattEccles(2, base_ring=ZZ)(((1, 2),))
    with pytest.raises(TypeError, match="same base ring"):
        BarrattEccles.compose(x, 1, y)


def test_barratt_eccles_constructor_keeps_degenerate_inputs_zero() -> None:
    e2 = BarrattEccles(2, QQ)
    assert e2(((1, 2), (1, 2))) == e2.zero()


@pytest.mark.parametrize(
    ("key", "error_type"),
    [
        (((1, 2, 3),), ValueError),
        (((1, 3),), ValueError),
        (((1, "2"),), TypeError),
    ],
)
def test_barratt_eccles_constructor_raises_on_invalid_basis_key(
    key: tuple[tuple[object, ...], ...], error_type: type[Exception]
) -> None:
    e2 = BarrattEccles(2, QQ)
    with pytest.raises(error_type):
        e2(key)


def test_barratt_eccles_constructor_raises_on_invalid_container() -> None:
    e2 = BarrattEccles(2, QQ)
    with pytest.raises(TypeError):
        e2("12")


def test_barratt_eccles_dict_constructor_skips_only_degenerate_terms() -> None:
    e2 = BarrattEccles(2, QQ)
    element = e2({((1, 2),): 3, ((1, 2), (1, 2)): 5})
    assert _as_dict(element) == {((2, 1),): 3}


def test_barratt_eccles_dict_constructor_raises_on_invalid_key() -> None:
    e2 = BarrattEccles(2, QQ)
    with pytest.raises(ValueError):
        e2({((1, 2),): 1, ((1, 3),): 2})


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_barratt_eccles_operadic_unit_axioms(input_pos: int) -> None:
    e3 = BarrattEccles(3, QQ)
    x = e3(((1, 2, 3),))
    one = BarrattEccles.unit(QQ)

    left = BarrattEccles.compose(one, 1, x)
    right = BarrattEccles.compose(x, input_pos, one)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed for Barratt-Eccles."
    assert _as_dict(right) == x_dict, f"Right unit axiom failed at input {input_pos}."
