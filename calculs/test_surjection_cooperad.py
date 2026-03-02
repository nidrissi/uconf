"""Regression tests for Surjection cooperad primitives."""

from uconf import Surjection, SurjectionLinearDual


def _canonical(x):
    if hasattr(x, "tuple"):
        return tuple(int(v) for v in x.tuple())
    if isinstance(x, tuple):
        return tuple(_canonical(v) for v in x)
    return x


def _as_dict(x):
    return {_canonical(k): int(v) for k, v in x if int(v) != 0}


def _tensor_coeff(delta, left_basis, right_basis) -> int:
    d = _as_dict(delta)
    return d.get((_canonical(left_basis), _canonical(right_basis)), 0)


def test_surjection_counit_unit_and_reduced() -> None:
    s1 = SurjectionLinearDual(1)
    unit = s1((1,))
    x = 3 * unit

    assert SurjectionLinearDual.counit(unit) == 1
    assert unit.counit() == 1
    assert SurjectionLinearDual.counit(x) == 3
    assert x.reduced() == s1.zero()


def test_surjection_counit_vanishes_outside_arity_one() -> None:
    x = SurjectionLinearDual(2)((1, 2))
    assert SurjectionLinearDual.counit(x) == 0
    assert x.reduced() == x


def test_infinitesimal_cocompose_transposes_compose_pairing() -> None:
    left = Surjection(2)((1, 2, 1))
    right = Surjection(2)((1, 2))
    i = 1

    composed = Surjection.compose(left, i, right)
    left_basis = next(iter(left.support()))
    right_basis = next(iter(right.support()))

    for u_basis, coeff in composed:
        u = SurjectionLinearDual(3)(u_basis)
        delta = u.infinitesimal_cocompose(i=i, m=2, n=2)
        assert _tensor_coeff(delta, left_basis, right_basis) == int(coeff)
