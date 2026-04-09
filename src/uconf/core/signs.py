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

    if hasattr(sigma, "tuple"):
        values = tuple(int(v) for v in sigma.tuple())
    else:
        values = tuple(int(v) for v in sigma)
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


def koszul_sign_of_permutation(perm_0idx: list[int], degrees: list[int]) -> int:
    """Compute the Koszul sign of a permutation acting on graded elements.

    Given a permutation perm and degrees [d_1, ..., d_k], compute the sign
    incurred by permuting elements of degrees d_{perm[0]}, d_{perm[1]}, ...
    back to their original order.

    The sign is (-1)^{sum of d_i * d_j for all inversions (i,j) in perm}.

    Warning: This function assumes that the input permutation is zero-indexed.
    """
    n = len(perm_0idx)
    exponent = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Inversion: perm_0idx[i] > perm_0idx[j]
            if perm_0idx[i] > perm_0idx[j]:
                exponent += degrees[perm_0idx[i]] * degrees[perm_0idx[j]]
    return 1 if exponent % 2 == 0 else -1


def get_on_basis(morphism: Any):
    """Extract the ``on_basis`` callable from a Sage module morphism.

    Returns the ``on_basis`` function if available, or ``None``.
    Calling ``on_basis(key)`` directly bypasses the overhead of
    ``morphism.__call__`` → ``linear_combination`` for single-term inputs.
    """
    fn = getattr(morphism, "on_basis", None)
    if fn is not None:
        fn = fn()
    return fn
