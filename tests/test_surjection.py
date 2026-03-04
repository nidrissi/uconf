"""Regression tests for the Surjection operad."""

from collections.abc import Iterable
import math

import pytest
from sage.all import ZZ

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


def usual_planar_test(rmax: int, dmax: int) -> Iterable[Surjection.Element]:
    for r in range(1, rmax):
        for d in range(1, dmax):
            yield from Surjection(r).planar_basis_it(d)


PLANAR_XSMALL = tuple(usual_planar_test(4, 2))
PLANAR_SMALL = tuple(usual_planar_test(4, 3))
PLANAR_LARGE = tuple(usual_planar_test(6, 3))


def test_surjection_unit() -> None:
    u = Surjection.unit()
    assert _as_dict(u) == {(1,): 1}, "Surjection unit should be (1,)."


@pytest.mark.parametrize("sigma", ([2, 1],))
def test_surjection_permutation_action(sigma: list[int]) -> None:
    s2 = Surjection(2)
    x = s2((1, 2))
    swapped = x.permute(sigma)
    assert _as_dict(swapped) == {(2, 1): 1}, "Expected transposed surjection tuple."
    assert _as_dict(swapped.permute(sigma)) == _as_dict(x), (
        "Applying the same transposition twice should return the original element."
    )


def test_surjection_basic_composition() -> None:
    s2 = Surjection(2)
    x = s2((1, 2))
    composed = Surjection.compose(x, 1, x)
    assert _as_dict(composed) == {(1, 2, 3): 1}, "Expected basis element (1,2,3)."


def test_surjection_complexity_uses_full_label_range() -> None:
    s2 = Surjection(2)
    x = s2((1, 2, 1))
    assert x.complexity() == 2


def test_surjection_compose_requires_same_base_ring() -> None:
    x = Surjection(2)((1, 2))
    y = Surjection(2, base_ring=ZZ)((1, 2))
    with pytest.raises(TypeError, match="same base ring"):
        Surjection.compose(x, 1, y)


@pytest.mark.parametrize("input_pos", [1, 2, 3])
def test_surjection_operadic_unit_axioms(input_pos: int) -> None:
    s3 = Surjection(3)
    x = s3((1, 2, 3))
    one = Surjection.unit()

    left = Surjection.compose(one, 1, x)
    right = Surjection.compose(x, input_pos, one)

    x_dict = _as_dict(x)
    assert _as_dict(left) == x_dict, "Left unit axiom failed for Surjection."
    assert _as_dict(right) == x_dict, f"Right unit axiom failed at input {input_pos}."


@pytest.mark.parametrize("r", range(1, 5))
@pytest.mark.parametrize("d", range(0, 5))
def test_surjection_basis(r: int, d: int) -> None:
    basis = list(Surjection(r).basis_it(d))
    assert len(basis) == len(set(basis)), "Duplicate elements found in basis_it"
    for el in basis:
        assert isinstance(el, Surjection.Element), (
            "Non-Surjection element found in basis_it"
        )
        assert el != el.parent().zero(), "Zero element found in basis_it"
        assert el.arity() == r, "Element with incorrect arity in basis_it"
        assert _degree_matches(el, d), (
            "Element with incorrect degree in Surjection basis_it"
        )


@pytest.mark.parametrize("r", range(2, 5))
@pytest.mark.parametrize("d", range(1, 4))
def test_planar_surjection_basis(r: int, d: int) -> None:
    planar_basis = list(Surjection(r).planar_basis_it(d))
    basis = list(Surjection(r).basis_it(d))
    assert set(planar_basis).issubset(basis), (
        "Planar basis should be a subset of the full basis."
    )
    assert len(planar_basis) * math.factorial(r) == len(basis), (
        "Planar basis size should be full basis size divided by r!."
    )

    for el in planar_basis:
        assert el.is_planar(), "Non-planar element found in planar_basis_it"


@pytest.mark.parametrize("s1", PLANAR_SMALL)
@pytest.mark.parametrize("s2", PLANAR_SMALL)
def test_planar_preserved_under_composition_last_input(
    s1: Surjection.Element, s2: Surjection.Element
) -> None:
    pos = s1.arity()
    composed = Surjection.compose(s1, pos, s2)
    assert composed.is_planar(), (
        f"Composition of {s1} and {s2} at position {pos} is not planar."
    )


@pytest.mark.parametrize("s", PLANAR_LARGE)
def test_section_right_inverse(s: Surjection.Element) -> None:
    sect = s.section()
    nat = sect.table_reduction()
    assert nat == s, f"Section failed for {s}, got {nat} instead."


@pytest.mark.parametrize("s", PLANAR_LARGE)
def test_section_planar(s: Surjection.Element) -> None:
    r = s.arity()
    sect = s.section()
    for key in sect.support():
        first_perm = key[0]
        assert tuple(first_perm.tuple()) == tuple(range(1, r + 1)), (
            f"Section of planar surjection {s} is not planar at {key}."
        )


@pytest.mark.skip(reason="The result is false")
@pytest.mark.parametrize("s1", PLANAR_XSMALL)
@pytest.mark.parametrize("s2", PLANAR_XSMALL)
def test_section_composition(s1: Surjection.Element, s2: Surjection.Element):
    """Check that the section map commutes with composition."""
    composed = Surjection.compose(s1, 1, s2)
    sect_composed = composed.section()

    sect1 = s1.section()
    sect2 = s2.section()
    composed_sect = Surjection.compose(sect1, 1, sect2)
    assert sect_composed == composed_sect, (
        f"Section composition failed for {s1} and {s2} at position 1."
    )


@pytest.mark.parametrize("i,j", [(1, 2), (1, 3), (2, 3)])
def test_surjection_parallel_composition_axiom_arity3(i: int, j: int) -> None:
    """Parallel composition axiom for p in S(3), q, r in S(2).

    For i < j: (p ∘_i q) ∘_{j + arity(q) - 1} r = (-1)^{|q|*|r|} * (p ∘_j r) ∘_i q
    """
    if i >= j:
        return
    p = Surjection(3)((1, 2, 3, 1))  # degree 1
    q = Surjection(2)((1, 2, 1))  # degree 1
    r = Surjection(2)((1, 2, 1))  # degree 1
    sign = (-1) ** (q.degree() * r.degree())
    lhs = Surjection.compose(Surjection.compose(p, i, q), j + q.arity() - 1, r)
    rhs = sign * Surjection.compose(Surjection.compose(p, j, r), i, q)
    assert lhs == rhs, f"Parallel axiom failed for i={i}, j={j}"


@pytest.mark.parametrize("i,j", [(1, 2), (1, 3), (2, 3)])
def test_surjection_parallel_composition_axiom_arity4(i: int, j: int) -> None:
    """Parallel composition axiom for p in S(4), q in S(3), r in S(2)."""
    if i >= j:
        return
    p = Surjection(4)((1, 2, 3, 4))  # degree 0
    q = Surjection(3)((1, 2, 3, 1))  # degree 1
    r = Surjection(2)((1, 2, 1))  # degree 1
    sign = (-1) ** (q.degree() * r.degree())
    lhs = Surjection.compose(Surjection.compose(p, i, q), j + q.arity() - 1, r)
    rhs = sign * Surjection.compose(Surjection.compose(p, j, r), i, q)
    assert lhs == rhs, f"Parallel axiom failed for i={i}, j={j}"


@pytest.mark.parametrize("i,j", [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
def test_surjection_parallel_composition_axiom_arity5(i: int, j: int) -> None:
    """Parallel composition axiom for p in S(5), q, r in S(2)."""
    if i >= j:
        return
    p = Surjection(5)((1, 2, 3, 4, 5))  # degree 0
    q = Surjection(2)((1, 2, 1))  # degree 1
    r = Surjection(2)((1, 2, 1))  # degree 1
    sign = (-1) ** (q.degree() * r.degree())
    lhs = Surjection.compose(Surjection.compose(p, i, q), j + q.arity() - 1, r)
    rhs = sign * Surjection.compose(Surjection.compose(p, j, r), i, q)
    assert lhs == rhs, f"Parallel axiom failed for i={i}, j={j}"


@pytest.mark.skip(reason="The result is false")
@pytest.mark.parametrize("s", PLANAR_SMALL)
def test_section_boundary(s: Surjection.Element):
    """Check that the section map commutes with the boundary map."""
    sect = s.section()
    sect_boundary = sect.boundary()
    s_boundary = s.boundary()
    sect_of_boundary = s_boundary.section()
    assert sect_boundary == sect_of_boundary, f"Section boundary failed for {s}."


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
    x = Surjection(m)(x_tuple)
    y = Surjection(n)(y_tuple)
    z = Surjection(p)(z_tuple)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            lhs = Surjection.compose(Surjection.compose(x, i, y), i + j - 1, z)
            rhs = Surjection.compose(x, i, Surjection.compose(y, j, z))
            assert lhs == rhs, (
                f"Sequential axiom failed for x={x_tuple}, y={y_tuple}, "
                f"z={z_tuple}, i={i}, j={j}"
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
    x = Surjection(m)(x_tuple)
    y = Surjection(n)(y_tuple)
    sigma_inv = _inverse_one_line(sigma)
    for i in range(1, m + 1):
        lhs = Surjection.compose(x.permute(sigma), i, y.permute(tau))
        block = _block_permutation(sigma, i, tau)
        rhs = Surjection.compose(x, sigma_inv[i - 1], y).permute(block)
        assert lhs == rhs, (
            f"Equivariance failed for x={x_tuple}, σ={sigma}, "
            f"y={y_tuple}, τ={tau}, i={i}"
        )


# ===========================================================================
# Square-zero differential:  d² = 0
# ===========================================================================


@pytest.mark.parametrize(
    "n,d",
    [
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 1),
    ],
)
def test_surjection_differential_squared_zero(n: int, d: int) -> None:
    """d²(x) = 0 for every degree-d basis element of S(n)."""
    zero = Surjection(n).zero()
    for elem in Surjection(n).basis_it(d):
        assert elem.boundary().boundary() == zero, (
            f"d²({elem}) ≠ 0 in S({n}) degree {d}"
        )


# ===========================================================================
# Derivation / Leibniz rule:  d(x∘_i y) = d(x)∘_i y + (-1)^|x| x∘_i d(y)
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,y_tuple,m,n",
    [
        # degree-1 × degree-0
        ((1, 2, 1), (1, 2), 2, 2),
        ((1, 2, 3, 1), (1, 2), 3, 2),
        ((1, 2, 3, 4, 1), (1, 2), 4, 2),
        # degree-0 × degree-1
        ((1, 2, 3), (1, 2, 1), 3, 2),
        ((1, 2, 3, 4), (1, 2, 1), 4, 2),
        ((1, 2, 3), (1, 2, 3, 1), 3, 3),
        # degree-1 × degree-1
        ((1, 2, 3, 1), (1, 2, 1), 3, 2),
        ((1, 2, 3, 4, 1), (1, 2, 1), 4, 2),
        # degree-2 × degree-0
        ((1, 2, 1, 2), (1, 2), 2, 2),
        ((1, 2, 3, 1, 2), (1, 2), 3, 2),
        # degree-0 × degree-2
        ((1, 2, 3), (1, 2, 1, 2), 3, 2),
        # degree-2 × degree-1
        ((1, 2, 1, 2), (1, 2, 1), 2, 2),
    ],
)
def test_surjection_derivation_property(
    x_tuple: tuple, y_tuple: tuple, m: int, n: int
) -> None:
    """d(x∘_i y) = d(x)∘_i y + (-1)^|x| x∘_i d(y) for all valid i."""
    x = Surjection(m)(x_tuple)
    y = Surjection(n)(y_tuple)
    sign = (-1) ** x.degree()
    for i in range(1, m + 1):
        xy = Surjection.compose(x, i, y)
        lhs = xy.boundary()
        rhs = Surjection.compose(x.boundary(), i, y) + sign * Surjection.compose(
            x, i, y.boundary()
        )
        assert lhs == rhs, f"Derivation failed for x={x_tuple}, y={y_tuple}, i={i}"
