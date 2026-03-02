"""Shared sign conventions for shifted operadic/cooperadic constructions."""

from __future__ import annotations

from typing import Any


def sign_from_exponent(exponent: int) -> int:
    """Return ``(-1)^exponent`` as ``+1`` or ``-1``."""

    return -1 if exponent % 2 else 1


def permutation_signature(sigma: Any) -> int:
    """Return the signature of a permutation as ``+1`` or ``-1``."""

    if hasattr(sigma, "signature"):
        return int(sigma.signature())
    if hasattr(sigma, "sign"):
        return int(sigma.sign())

    values = tuple(int(v) for v in sigma.tuple())
    inversions = 0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if left > right:
                inversions += 1
    return -1 if inversions % 2 else 1


def shifted_permutation_sign(shift_degree: int, sigma: Any) -> int:
    """Return the suspension twist ``sgn(sigma)^shift_degree``."""

    return permutation_signature(sigma) ** int(shift_degree)


def shifted_boundary_sign(shift_degree: int) -> int:
    """Return the differential transport sign for an integer shift."""

    return sign_from_exponent(int(shift_degree))


def shifted_operadic_compose_sign(
    shift_degree: int,
    input_position: int,
    left_arity: int,
    right_arity: int,
    right_degree: int,
) -> int:
    """Return shift sign for operadic partial composition.

    Formula used in :class:`uconf.shifted_operad.ShiftedOperad`:
    ``(-1)^(d * ((i-1)(n-1) + (m-1)|y|))``.
    """

    exponent = int(shift_degree) * (
        (int(input_position) - 1) * (int(right_arity) - 1)
        + (int(left_arity) - 1) * int(right_degree)
    )
    return sign_from_exponent(exponent)
