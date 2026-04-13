"""Regression tests for the Barratt-Eccles operad."""

import itertools
import math
from random import Random

import pytest
from sage.all import ZZ, QQ, SymmetricGroup

from uconf import BarrattEccles
from tests.planarize_helpers import planarize_round_trip_ok


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


def _random_coeff(rng: Random) -> int:
    coeff = 0
    while coeff == 0:
        coeff = rng.randint(-2, 2)
    return coeff


def _random_homogeneous_be(
    rng: Random, arity: int, degree: int, max_terms: int = 3
) -> BarrattEccles.Element:
    parent = BarrattEccles(arity, QQ)
    basis = list(parent.basis_iter(degree))
    assert basis, f"No basis for BarrattEccles({arity}, QQ) in degree {degree}."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def test_stress_barratt_eccles_linearity_and_unit() -> None:
    rng = Random(20260228)

    for _ in range(10):
        m = rng.randint(2, 3)
        n = rng.randint(2, 3)
        dx = rng.randint(0, 1)
        dz = rng.randint(0, 1)

        x = _random_homogeneous_be(rng, m, dx)
        y = _random_homogeneous_be(rng, m, dx)
        z = _random_homogeneous_be(rng, n, dz)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = BarrattEccles.compose(a * x + b * y, i, z)
        rhs = a * BarrattEccles.compose(x, i, z) + b * BarrattEccles.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = BarrattEccles.unit(QQ)
        assert _as_dict(BarrattEccles.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(BarrattEccles.compose(x, k, one)) == _as_dict(x)


def test_stress_barratt_eccles_boundary_squared_zero() -> None:
    rng = Random(20260303)

    for _ in range(15):
        r = rng.randint(2, 4)
        d = rng.randint(0, 2)
        x = _random_homogeneous_be(rng, r, d)
        assert _as_dict(x.boundary().boundary()) == {}


@pytest.mark.parametrize("n", range(2, 4))
@pytest.mark.parametrize("d", range(0, 3))
def test_barratt_eccles_right_action(n: int, d: int) -> None:
    """(x·σ)·τ = x·(στ) for all σ, τ ∈ S(n)."""
    BEn = BarrattEccles(n, QQ)
    Sn = BEn._symmetric_group()
    for x in BEn.basis_iter(d):
        for sigma, tau in itertools.product(list(Sn), repeat=2):
            lhs = x.permute(sigma).permute(tau)
            rhs = x.permute(sigma * tau)
            assert lhs == rhs, f"Right action failed for x={x}, σ={sigma}, τ={tau}"


@pytest.mark.parametrize("n", range(2, 4))
@pytest.mark.parametrize("d", range(0, 3))
def test_barratt_eccles_planarize_round_trip(n: int, d: int) -> None:
    """Planarize round-trip holds for homogeneous Barratt-Eccles elements."""
    BEn = BarrattEccles(n, QQ)
    for x in BEn.basis_iter(d):
        assert planarize_round_trip_ok(x), f"Planarize round-trip failed for {x}"


@pytest.mark.parametrize("r", range(2, 5))
@pytest.mark.parametrize("d", range(0, 4))
def test_table_reduction_equivariant(r: int, d: int) -> None:
    Sr = SymmetricGroup(r)
    for x in BarrattEccles(r, QQ).planar_basis_iter(d):
        for sigma in Sr:
            permuted_of_table = x.permute(sigma).table_reduction()
            reduced_of_permuted = x.table_reduction().permute(sigma)
            assert _as_dict(permuted_of_table) == _as_dict(reduced_of_permuted), (
                f"Section failed for {permuted_of_table}, got {reduced_of_permuted} instead."
            )
