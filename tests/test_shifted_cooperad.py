"""Tests for the shifted-cooperad wrapper."""

from uconf import ShiftedCooperad, SurjectionDual
from sage.all import QQ


def _as_dict(x):
    return {basis: int(coeff) for basis, coeff in x if int(coeff) != 0}


def test_degree_shift_surjection_linear_dual() -> None:
    shifted = ShiftedCooperad(SurjectionDual, 1)
    c2 = shifted(2, QQ)
    x = c2((1, 2, 1))
    basis = next(iter(x.support()))
    assert c2.degree_on_basis(basis) == 0


def test_counit_preserved_by_shift_wrapper() -> None:
    shifted = ShiftedCooperad(SurjectionDual, 1)
    x = shifted(1, QQ)((1,))
    assert x.counit() == 1


def test_shifted_infinitesimal_cocompose_sign() -> None:
    base = SurjectionDual(3, QQ)((1, 2, 3, 2))
    shifted = ShiftedCooperad(SurjectionDual, 1)
    shifted_x = shifted(3, QQ)((1, 2, 3, 2))

    base_delta = SurjectionDual.infinitesimal_cocompose(base, i=1, m=2, n=2)
    shifted_delta = shifted_x.infinitesimal_cocompose(i=1, m=2, n=2)

    b = _as_dict(base_delta)
    s = _as_dict(shifted_delta)

    assert b.keys() == s.keys()
    right_parent = SurjectionDual(2, QQ)
    for (left_basis, right_basis), coeff in b.items():
        right_degree = right_parent.degree_on_basis(right_basis)
        expected_sign = -1 if right_degree % 2 else 1
        assert s[(left_basis, right_basis)] == expected_sign * coeff


def test_element_repr_latex_exists() -> None:
    shifted = ShiftedCooperad(SurjectionDual, 1)
    x = shifted(2, QQ)((1, 2, 1))
    ltx = x._repr_latex_()
    assert ltx
