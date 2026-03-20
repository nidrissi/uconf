"""Tests for the shifted-operad wrapper."""

import pytest
from sage.all import ZZ, QQ

from uconf import Lie, ShiftedOperad, Surjection


def _as_dict(x):
    return {basis: coeff for basis, coeff in x}


def test_degree_shift_lie() -> None:
    shifted = ShiftedOperad(Lie, 2)
    l3 = shifted(3, ZZ)
    x = l3((1, 2))
    basis = next(iter(x.support()))
    assert l3.degree_on_basis(basis) == 4


def test_degree_shift_surjection() -> None:
    shifted = ShiftedOperad(Surjection, 1)
    s2 = shifted(2, ZZ)
    x = s2((1, 2, 1))
    basis = next(iter(x.support()))
    assert s2.degree_on_basis(basis) == 2


def test_permutation_sign_twist_on_lie() -> None:
    odd_shift = ShiftedOperad(Lie, 1)
    even_shift = ShiftedOperad(Lie, 2)

    x_odd = odd_shift(2, ZZ)((1,))
    x_even = even_shift(2, ZZ)((1,))

    transposition = [2, 1]
    assert x_odd.permute(transposition) == x_odd
    assert x_even.permute(transposition) == -x_even


def test_composition_sign_shift_lie() -> None:
    shifted = ShiftedOperad(Lie, 1)
    x = shifted(2, QQ)((1,))
    assert x.arity() == 2
    assert x.degree() == 1

    composed = shifted.compose(x, 2, x)
    assert composed.arity() == 3
    assert composed.degree() == 2

    base_composed = Lie.compose(Lie(2, QQ)((1,)), 2, Lie(2, QQ)((1,)))

    assert _as_dict(composed.base_element()) == _as_dict(-base_composed)


def test_composition_sign_shift_surjection_nonzero_degree() -> None:
    shifted = ShiftedOperad(Surjection, 1)
    x = shifted(2, QQ)((1, 2))
    y = shifted(2, QQ)((1, 2, 1))

    composed = shifted.compose(x, 1, y)
    base_composed = Surjection.compose(Surjection(2, QQ)((1, 2)), 1, Surjection(2, QQ)((1, 2, 1)))

    assert _as_dict(composed.base_element()) == _as_dict(-base_composed)


def test_compose_requires_same_base_ring() -> None:
    shifted = ShiftedOperad(Surjection, 1)
    x = shifted(2, QQ)((1, 2))
    y = shifted(2, base_ring=ZZ)((1, 2))
    with pytest.raises(TypeError, match="same base ring"):
        shifted.compose(x, 1, y)


def test_readme_example_smoke() -> None:
    shift_lie = ShiftedOperad(Lie, 1)
    l2 = shift_lie(2, QQ)
    x = l2((1,))

    assert x.permute([2, 1]) == x

    z = shift_lie.compose(x, 2, x)
    expected = -Lie.compose(Lie(2, QQ)((1,)), 2, Lie(2, QQ)((1,)))
    assert _as_dict(z.base_element()) == _as_dict(expected)


def test_basis_it_negative_shift() -> None:
    """ShiftedOperad(Lie, -1)(n) has elements in degree -(n-1)."""
    sLie = ShiftedOperad(Lie, -1)

    # arity 2: degree = 0 + (-1)*(2-1) = -1
    sLie2 = sLie(2, QQ)
    assert list(sLie2.basis_it(-1)) != []
    assert list(sLie2.basis_it(0)) == []

    # arity 3: degree = 0 + (-1)*(3-1) = -2
    sLie3 = sLie(3, QQ)
    assert list(sLie3.basis_it(-2)) != []
    assert list(sLie3.basis_it(-1)) == []
    assert list(sLie3.basis_it(0)) == []


def test_basis_it_positive_shift() -> None:
    """ShiftedOperad(Lie, 1)(n) has elements in degree n-1."""
    sLie = ShiftedOperad(Lie, 1)

    # arity 2: degree = 0 + 1*(2-1) = 1
    sLie2 = sLie(2, QQ)
    assert list(sLie2.basis_it(1)) != []
    assert list(sLie2.basis_it(0)) == []

    # arity 3: degree = 0 + 1*(3-1) = 2
    sLie3 = sLie(3, QQ)
    assert list(sLie3.basis_it(2)) != []
    assert list(sLie3.basis_it(1)) == []


def test_basis_it_keys_match_degree_on_basis() -> None:
    """Every key yielded by basis_it(d) satisfies degree_on_basis == d."""
    sLie = ShiftedOperad(Lie, -1)
    for n in (2, 3):
        comp = sLie(n, QQ)
        d = -(n - 1)
        for elem in comp.basis_it(d):
            for key, _ in elem:
                assert comp.degree_on_basis(key) == d
