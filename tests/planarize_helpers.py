"""Shared test helpers for quasi-planar round-trip checks."""

from sage.all import SymmetricGroup


def planarize_round_trip_ok(element) -> bool:
    """Return whether ``sum c_i * p_i.permute(sigma_i) == element`` after planarize.

    This validates the standard quasi-planar identity for any test element whose
    parent supports ``planarize`` and ``permute``.
    """
    parent = element.parent()
    if element == parent.zero():
        return True

    if not callable(getattr(element, "planarize", None)):
        raise AttributeError(f"{type(element).__name__} does not define planarize")

    total = parent.zero()
    symmetric_group = SymmetricGroup(parent.arity())
    for (planar_key, sigma_key), coeff in element.planarize():
        sigma = symmetric_group(sigma_key)
        total += coeff * parent(planar_key).permute(sigma)

    return total == element
