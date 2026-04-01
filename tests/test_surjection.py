"""Regression tests for the Surjection operad."""

import math
from random import Random

import pytest
from sage.all import ZZ, QQ

from uconf import Surjection


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


def test_surjection_unit() -> None:
    u = Surjection.unit(QQ)
    assert _as_dict(u) == {(1,): 1}, "Surjection unit should be (1,)."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_surjection_permutation_action(sigma: list[int]) -> None:
    s2 = Surjection(2, QQ)
    x = s2((1, 2))
    swapped = x.permute(sigma)
    assert _as_dict(swapped) == {(2, 1): 1}, "Expected transposed surjection tuple."
    assert _as_dict(swapped.permute(sigma)) == _as_dict(x), (
        "Applying the same transposition twice should return the original element."
    )


def test_surjection_basic_composition() -> None:
    s2 = Surjection(2, QQ)
    x = s2((1, 2))
    composed = Surjection.compose(x, 1, x)
    assert _as_dict(composed) == {(1, 2, 3): 1}, "Expected basis element (1,2,3)."


def test_surjection_complexity_uses_full_label_range() -> None:
    s2 = Surjection(2, QQ)
    x = s2((1, 2, 1))
    assert x.complexity() == 2


def test_surjection_compose_requires_same_base_ring() -> None:
    x = Surjection(2, QQ)((1, 2))
    y = Surjection(2, base_ring=ZZ)((1, 2))
    with pytest.raises(TypeError, match="same base ring"):
        Surjection.compose(x, 1, y)


def test_surjection_constructor_keeps_degenerate_inputs_zero() -> None:
    s2 = Surjection(2, QQ)
    assert s2((1, 1, 2)) == s2.zero()
    assert s2((1, 1, 1)) == s2.zero()


@pytest.mark.parametrize(
    ("key", "error_type"),
    [
        ((0, 1), ValueError),
        ((1, 3), ValueError),
        ((1, "2"), TypeError),
    ],
)
def test_surjection_constructor_raises_on_invalid_basis_key(
    key: tuple[object, ...], error_type: type[Exception]
) -> None:
    s2 = Surjection(2, QQ)
    with pytest.raises(error_type):
        s2(key)


def test_surjection_constructor_raises_on_invalid_container() -> None:
    s2 = Surjection(2, QQ)
    with pytest.raises(TypeError):
        s2("12")


def test_surjection_dict_constructor_skips_only_degenerate_terms() -> None:
    s2 = Surjection(2, QQ)
    element = s2({(1, 2): 3, (1, 1, 2): 5})
    assert _as_dict(element) == {(1, 2): 3}


def test_surjection_dict_constructor_raises_on_invalid_key() -> None:
    s2 = Surjection(2, QQ)
    with pytest.raises(ValueError):
        s2({(1, 2): 1, (1, 3): 2})


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_surjection_operadic_unit_axioms(input_pos: int) -> None:
    s3 = Surjection(3, QQ)
    x = s3((1, 2, 3))
    one = Surjection.unit(QQ)

    left = Surjection.compose(one, 1, x)
    right = Surjection.compose(x, input_pos, one)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed for Surjection."
    assert _as_dict(right) == x_dict, f"Right unit axiom failed at input {input_pos}."


@pytest.mark.parametrize("r", range(1, 3))
@pytest.mark.parametrize("d", range(0, 3))
def test_surjection_basis(r: int, d: int) -> None:
    basis = list(Surjection(r, QQ).basis_iter(d))
    assert len(basis) == len(set(basis)), "Duplicate elements found in basis_iter"
    for el in basis:
        assert isinstance(el, Surjection.Element), "Non-Surjection element found in basis_iter"
        assert el != el.parent().zero(), "Zero element found in basis_iter"
        assert el.arity() == r, "Element with incorrect arity in basis_iter"
        assert _degree_matches(el, d), "Element with incorrect degree in Surjection basis_iter"


@pytest.mark.parametrize("r", range(2, 3))
@pytest.mark.parametrize("d", range(1, 3))
def test_planar_surjection_basis(r: int, d: int) -> None:
    planar_basis = list(Surjection(r, QQ).planar_basis_iter(d))
    basis = list(Surjection(r, QQ).basis_iter(d))
    assert set(planar_basis).issubset(basis), "Planar basis should be a subset of the full basis."
    assert len(planar_basis) * math.factorial(r) == len(basis), (
        "Planar basis size should be full basis size divided by r!."
    )

    for el in planar_basis:
        assert el.is_planar(), "Non-planar element found in planar_basis_iter"


@pytest.mark.parametrize("r1", range(2, 3))
@pytest.mark.parametrize("r2", range(2, 3))
@pytest.mark.parametrize("d1", range(0, 3))
@pytest.mark.parametrize("d2", range(0, 3))
def test_planar_preserved_under_composition_last_input(r1: int, d1: int, r2: int, d2: int) -> None:
    for s1 in Surjection(r1, QQ).planar_basis_iter(d1):
        for s2 in Surjection(r2, QQ).planar_basis_iter(d2):
            pos = s1.arity()
            composed = Surjection.compose(s1, pos, s2)
            assert composed.is_planar(), (
                f"Composition of {s1} and {s2} at position {pos} is not planar."
            )


@pytest.mark.parametrize("r", range(2, 4))
@pytest.mark.parametrize("d", range(0, 4))
def test_section_right_inverse(r: int, d: int) -> None:
    for s in Surjection(r, QQ).planar_basis_iter(d):
        sect = s.section()
        nat = sect.table_reduction()
        assert nat == s, f"Section failed for {s}, got {nat} instead."


@pytest.mark.parametrize("r", range(2, 4))
@pytest.mark.parametrize("d", range(0, 4))
def test_section_planar(r: int, d: int) -> None:
    for s in Surjection(r, QQ).planar_basis_iter(d):
        r = s.arity()
        sect = s.section()
        for key in sect.support():
            first_perm = key[0]
            assert tuple(first_perm.tuple()) == tuple(range(1, r + 1)), (
                f"Section of planar surjection {s} is not planar at {key}."
            )


@pytest.mark.parametrize(
    "p_tuple,q_tuple,r_tuple,i,j",
    [
        # p ∈ S(3), q = r = (1,2,1) ∈ S(2), degree 1 × 1
        ((1, 2, 3, 1), (1, 2, 1), (1, 2, 1), 1, 2),
        ((1, 2, 3, 1), (1, 2, 1), (1, 2, 1), 1, 3),
        ((1, 2, 3, 1), (1, 2, 1), (1, 2, 1), 2, 3),
        # p = (1,2,3,4) ∈ S(4), q = (1,2,3,1) ∈ S(3), r = (1,2,1) ∈ S(2)
        ((1, 2, 3, 4), (1, 2, 3, 1), (1, 2, 1), 1, 2),
        ((1, 2, 3, 4), (1, 2, 3, 1), (1, 2, 1), 1, 3),
        ((1, 2, 3, 4), (1, 2, 3, 1), (1, 2, 1), 2, 3),
        # p = (1,2,3,4,5) ∈ S(5), q = r = (1,2,1) ∈ S(2)
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 1, 2),
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 1, 3),
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 1, 4),
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 2, 3),
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 2, 4),
        ((1, 2, 3, 4, 5), (1, 2, 1), (1, 2, 1), 3, 4),
    ],
)
def test_surjection_parallel_composition_axiom(
    p_tuple: tuple,
    q_tuple: tuple,
    r_tuple: tuple,
    i: int,
    j: int,
) -> None:
    """Parallel composition axiom for i < j.

    For i < j: (p ∘_i q) ∘_{j + arity(q) - 1} r = (-1)^{|q|*|r|} * (p ∘_j r) ∘_i q
    """
    p = Surjection(max(p_tuple), QQ)(p_tuple)
    q = Surjection(max(q_tuple), QQ)(q_tuple)
    r = Surjection(max(r_tuple), QQ)(r_tuple)
    sign = (-1) ** (q.degree() * r.degree())
    lhs = Surjection.compose(Surjection.compose(p, i, q), j + q.arity() - 1, r)
    rhs = sign * Surjection.compose(Surjection.compose(p, j, r), i, q)
    assert lhs == rhs, f"Parallel axiom failed for i={i}, j={j}"


# ===========================================================================
# Sequential composition axiom:  (x∘_i y)∘_{i+j-1} z = x∘_i (y∘_j z)
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,y_tuple,z_tuple,m,n,p",
    [
        # degree-1 × degree-0 × degree-0
        ((1, 2, 1), (1, 2), (1, 2), 2, 2, 2),
        # degree-0 × degree-1 × degree-1
        ((1, 2, 3), (1, 2, 1), (1, 2, 1), 3, 2, 2),
        # degree-0 × degree-0 × degree-0, larger arities
        ((1, 2, 3, 4), (1, 2, 3), (1, 2), 4, 3, 2),
        ((1, 2, 3, 4), (1, 2), (1, 2, 3), 4, 2, 3),
        # degree-1 × degree-0 × degree-1
        ((1, 2, 3, 1), (1, 2), (1, 2, 1), 3, 2, 2),
        ((1, 2, 3, 4, 1), (1, 2), (1, 2, 1), 4, 2, 2),
        # degree-2 × degree-0 × degree-0
        ((1, 2, 1, 2), (1, 2), (1, 2), 2, 2, 2),
        # degree-0 × degree-1 × degree-0
        ((1, 2, 3), (1, 2, 1), (1, 2), 3, 2, 2),
        ((1, 2, 3, 4), (1, 2, 3, 1), (1, 2), 4, 3, 2),
    ],
)
def test_surjection_sequential_composition_axiom(
    x_tuple: tuple, y_tuple: tuple, z_tuple: tuple, m: int, n: int, p: int
) -> None:
    """(x∘_i y)∘_{i+j-1} z = x∘_i(y∘_j z) for all valid i, j."""
    x = Surjection(m, QQ)(x_tuple)
    y = Surjection(n, QQ)(y_tuple)
    z = Surjection(p, QQ)(z_tuple)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            lhs = Surjection.compose(Surjection.compose(x, i, y), i + j - 1, z)
            rhs = Surjection.compose(x, i, Surjection.compose(y, j, z))
            assert lhs == rhs, (
                f"Sequential axiom failed for x={x_tuple}, y={y_tuple}, z={z_tuple}, i={i}, j={j}"
            )


# ===========================================================================
# Equivariance:  (σ·x)∘_i(τ·y) = (σ∘_iτ) · (x∘_{σ^{-1}(i)} y)
# ===========================================================================


def _block_permutation(sigma: list[int], i: int, tau: list[int]) -> list[int]:
    """Return the block permutation ``σ ∘_i τ`` in one-line notation."""
    shift = len(tau) - 1
    result = []
    for v in sigma:
        if v < i:
            result.append(v)
        elif v > i:
            result.append(v + shift)
        else:
            result.extend([t + i - 1 for t in tau])
    return result


def _inverse_one_line(sigma: list[int]) -> list[int]:
    """Return inverse of a permutation given in one-line notation."""
    inv = [0] * len(sigma)
    for pos, val in enumerate(sigma, start=1):
        inv[val - 1] = pos
    return inv


@pytest.mark.parametrize(
    "x_tuple,sigma,y_tuple,tau,m,n",
    [
        # S(2)×S(2), σ=τ=(2,1), degree 0×0
        ((1, 2), [2, 1], (1, 2), [2, 1], 2, 2),
        # S(2)×S(2), degree 1×0
        ((1, 2, 1), [2, 1], (1, 2), [2, 1], 2, 2),
        # S(3)×S(2), σ=(2,3,1), τ=(2,1), degree 0×0
        ((1, 2, 3), [2, 3, 1], (1, 2), [2, 1], 3, 2),
        # S(3)×S(2), degree 1×0
        ((1, 2, 3, 1), [2, 3, 1], (1, 2), [2, 1], 3, 2),
        # S(3)×S(2), σ=(2,1,3), degree 0×1
        ((1, 2, 3), [2, 1, 3], (1, 2, 1), [2, 1], 3, 2),
        # S(4)×S(2), σ=(2,1,3,4), τ=(2,1), degree 0×0
        ((1, 2, 3, 4), [2, 1, 3, 4], (1, 2), [2, 1], 4, 2),
        # S(4)×S(2), degree 1×1
        ((1, 2, 3, 4, 1), [2, 1, 3, 4], (1, 2, 1), [2, 1], 4, 2),
        # S(3)×S(3), σ=(3,1,2), τ=(2,3,1), degree 0×0
        ((1, 2, 3), [3, 1, 2], (1, 2, 3), [2, 3, 1], 3, 3),
        # S(4)×S(3), σ=(2,3,1,4), τ=(3,1,2), degree 0×0
        ((1, 2, 3, 4), [2, 3, 1, 4], (1, 2, 3), [3, 1, 2], 4, 3),
    ],
)
def test_surjection_equivariance(
    x_tuple: tuple, sigma: list[int], y_tuple: tuple, tau: list[int], m: int, n: int
) -> None:
    """(σ·x)∘_i(τ·y) = (σ∘_iτ)·(x∘_{σ^{-1}(i)} y) for all valid i."""
    x = Surjection(m, QQ)(x_tuple)
    y = Surjection(n, QQ)(y_tuple)
    sigma_inv = _inverse_one_line(sigma)
    for i in range(1, m + 1):
        lhs = Surjection.compose(x.permute(sigma), i, y.permute(tau))
        block = _block_permutation(sigma, i, tau)
        rhs = Surjection.compose(x, sigma_inv[i - 1], y).permute(block)
        assert lhs == rhs, (
            f"Equivariance failed for x={x_tuple}, σ={sigma}, y={y_tuple}, τ={tau}, i={i}"
        )


# ===========================================================================
# Square-zero differential:  d² = 0
# ===========================================================================


def test_surjection_differential_squared_zero() -> None:
    """d²(x) = 0 for every degree-d basis element of S(n)."""
    rng = Random(20260324)
    for _ in range(20):
        n = rng.randint(2, 4)
        d = rng.randint(2, 5)
        zero = Surjection(n, QQ).zero()
        elem = rng.choice(Surjection(n, QQ).graded_basis(d))
        assert elem.boundary().boundary() == zero, f"d²({elem}) ≠ 0 in S({n}) degree {d}"


# ===========================================================================
# Derivation / Leibniz rule:  d(x∘_i y) = d(x)∘_i y + (-1)^|x| x∘_i d(y)
# ===========================================================================


def test_surjection_derivation_property() -> None:
    """d(x∘_i y) = d(x)∘_i y + (-1)^|x| x∘_i d(y) for all valid i."""
    rng = Random(20260311)
    for _ in range(50):
        n1 = rng.randint(2, 5)
        n2 = rng.randint(2, 5)
        d1 = rng.randint(1, 3)
        d2 = rng.randint(1, 3)
        x = rng.choice(Surjection(n1, QQ).graded_basis(d1))
        y = rng.choice(Surjection(n2, QQ).graded_basis(d2))
        sign = (-1) ** d1
        for i in range(1, n1 + 1):
            xy = Surjection.compose(x, i, y)
            lhs = xy.boundary()
            rhs = Surjection.compose(x.boundary(), i, y) + sign * Surjection.compose(
                x, i, y.boundary()
            )
            assert lhs == rhs, f"Derivation failed for x={x}, y={y}, i={i}"


def _random_coeff(rng: Random) -> int:
    coeff = 0
    while coeff == 0:
        coeff = rng.randint(-2, 2)
    return coeff


def _random_homogeneous_surjection(
    rng: Random, arity: int, degree: int, max_terms: int = 3
) -> Surjection.Element:
    parent = Surjection(arity, QQ)
    basis = list(parent.basis_iter(degree))
    assert basis, f"No basis for Surjection({arity}, QQ) in degree {degree}."
    k = min(max_terms, len(basis))
    picks = rng.sample(basis, k=k)
    data = {}
    for elt in picks:
        for b, _ in elt:
            data[b] = _random_coeff(rng)
    return parent.sum_of_terms(data.items())


def test_stress_surjection_linearity_and_unit() -> None:
    rng = Random(20260227)

    for _ in range(12):
        m = rng.randint(2, 3)
        n = rng.randint(2, 3)
        dx = rng.randint(0, 1)
        dz = rng.randint(0, 1)

        x = _random_homogeneous_surjection(rng, m, dx)
        y = _random_homogeneous_surjection(rng, m, dx)
        z = _random_homogeneous_surjection(rng, n, dz)

        i = rng.randint(1, m)

        a = rng.randint(-2, 2)
        b = rng.randint(-2, 2)
        lhs = Surjection.compose(a * x + b * y, i, z)
        rhs = a * Surjection.compose(x, i, z) + b * Surjection.compose(y, i, z)
        assert _as_dict(lhs) == _as_dict(rhs)

        one = Surjection.unit(QQ)
        assert _as_dict(Surjection.compose(one, 1, x)) == _as_dict(x)
        k = rng.randint(1, m)
        assert _as_dict(Surjection.compose(x, k, one)) == _as_dict(x)


def test_stress_surjection_boundary_squared_zero() -> None:
    rng = Random(20260302)

    for _ in range(20):
        r = rng.randint(2, 4)
        d = rng.randint(0, 2)
        x = _random_homogeneous_surjection(rng, r, d)
        assert _as_dict(x.boundary().boundary()) == {}
