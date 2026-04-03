"""Canonical operadic twisting morphisms.

This module provides the two fundamental twisting morphisms in operad theory:

- ``canonical_projection(P)``: the projection π: B(P) → P, which projects a
  bar tree to the operad element at its root (if it is a single-vertex corolla)
  and zero otherwise.

- ``canonical_inclusion(C)``: the inclusion ι: C → Ω(C), which sends an
  element c ∈ C(n) to the single-vertex cobar tree decorated by c.

These induce adjunctions:

- π gives B_π: P-alg → B(P)-coalg  (the standard bar construction for algebras)
  and Ω_π: B(P)-coalg → P-alg  (the standard cobar construction for coalgebras)

- ι gives Ω_ι: C-coalg → Ω(C)-alg  (the standard cobar construction for coalgebras)
  and B_ι: Ω(C)-alg → C-coalg  (the standard twisted bar construction)

For specific operad/cooperad pairs:

- ``canonical_projection(Ω(C))``: π: B(Ω(C)) → Ω(C), for a cooperad C
- ``canonical_inclusion(B(P))``: ι: B(P) → Ω(B(P)), for an operad P

Reference: Loday-Vallette "Algebraic Operads", Section 6.5 and 11.3.
"""

from __future__ import annotations

from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.cooperad import CooperadLike
from uconf.core.operad import OperadLike
from uconf.core.trees import (
    RootedTree,
    children,
    decoration,
    is_leaf,
    vertex_arity,
)
from uconf.core.twisting import TwistingMorphism


def canonical_projection(operad: OperadLike) -> TwistingMorphism:
    """Return the canonical projection π: B(P) → P.

    The projection sends a bar tree to:
    - The P-decoration of its root, if the tree is a single-vertex corolla
      (root vertex with all children being leaves).
    - Zero otherwise.

    For multi-vertex trees, the result is zero because the projection
    only sees the cogenerators of the cofree cooperad B(P).

    The sign convention is: π sends [p] (the corolla suspended by +1) to p,
    with no additional sign.

    Args:
        operad: A connected dg-operad P.

    Returns:
        A ``TwistingMorphism`` representing π: B(P) → P.

    Example::

        from uconf import Associative
        pi = canonical_projection(Associative)
        # pi.cooperad == BarConstruction(Associative)
        # pi.operad == Associative
    """
    bar_P = BarConstruction(operad)

    def _projection_fn(c_elem):
        """Project a B(P) element to P: keep only single-vertex corollas."""
        n = c_elem.arity()
        base_ring = c_elem.parent().base_ring()
        p_parent = operad(n, base_ring)
        result = p_parent.zero()

        for tree_key, coeff in c_elem:
            # Arity-1 elements (unit tree) map to zero (reduced cooperad)
            if is_leaf(tree_key):
                continue
            # Check if tree_key is a single-vertex corolla
            v_arity = vertex_arity(tree_key)
            if v_arity != n:
                continue
            chs = children(tree_key)
            if not all(is_leaf(c) for c in chs):
                continue
            # Single-vertex corolla: extract P-decoration
            dec = decoration(tree_key)
            result += coeff * p_parent(dec)

        return result

    return TwistingMorphism(
        cooperad=bar_P,
        operad=operad,
        morphism_fn=_projection_fn,
        name=f"π: B({getattr(operad, 'name', '?')}) → {getattr(operad, 'name', '?')}",
    )


def canonical_inclusion(cooperad: CooperadLike) -> TwistingMorphism:
    """Return the canonical inclusion ι: C → Ω(C).

    The inclusion sends an element c ∈ C(n) (for n ≥ 2) to the single-vertex
    cobar tree in Ω(C)(n) decorated by c:

        ι(c) = (c, 1, 2, ..., n)   in Ω(C)(n)

    For n = 1 (the coaugmentation), the map is zero.

    Args:
        cooperad: A connected dg-cooperad C.

    Returns:
        A ``TwistingMorphism`` representing ι: C → Ω(C).

    Example::

        from uconf import CoAssociative
        iota = canonical_inclusion(CoAssociative)
        # iota.cooperad == CoAssociative
        # iota.operad == CobarConstruction(CoAssociative)
    """
    cobar_C = CobarConstruction(cooperad)

    def _inclusion_fn(c_elem):
        """Include a C element as a single-vertex cobar tree in Ω(C)."""
        n = c_elem.arity()
        base_ring = c_elem.parent().base_ring()
        cobar_parent = cobar_C(n, base_ring)

        if n <= 1:
            return cobar_parent.zero()

        result = cobar_parent.zero()
        for c_key, coeff in c_elem:
            # Build single-vertex cobar tree: (c_key, 1, 2, ..., n)
            cobar_tree_key = RootedTree(c_key, *range(1, n + 1))
            result += coeff * cobar_parent(cobar_tree_key)

        return result

    return TwistingMorphism(
        cooperad=cooperad,
        operad=cobar_C,
        morphism_fn=_inclusion_fn,
        name=f"ι: {getattr(cooperad, 'name', '?')} → Ω({getattr(cooperad, 'name', '?')})",
    )
