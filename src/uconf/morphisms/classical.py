"""Classical operad morphisms.

Provides the standard morphisms between classical operads:

- ``ass_to_com``: ``Ass → Com`` (augmentation — sends every permutation to
  the commutative generator).
- ``lie_to_ass``: ``Lie → Ass`` (PBW inclusion — sends a Lie bracket to
  its commutator expansion in the associative operad).
"""

from __future__ import annotations

from typing import Any

from uconf.core.morphism import OperadMorphism
from uconf.models.associative import Associative
from uconf.models.commutative import Commutative
from uconf.models.lie import Lie


def _ass_to_com_on_element(element: Any) -> Any:
    """Map an element of ``Ass(n)`` to ``Com(n)`` by summing coefficients."""
    n = element.arity()
    base_ring = element.parent().base_ring()
    target = Commutative(n, base_ring)
    total_coeff = sum(coeff for _, coeff in element)
    if total_coeff == 0:
        return target.zero()
    return total_coeff * target(())


ass_to_com = OperadMorphism(Associative, Commutative, _ass_to_com_on_element)
"""Augmentation morphism ``Ass → Com``.

Every permutation ``σ ∈ Ass(n)`` maps to the unique generator of ``Com(n)``.
"""


def _lie_to_ass_on_element(element: Any) -> Any:
    """Map an element of ``Lie(n)`` to ``Ass(n)`` via the PBW expansion."""
    parent = element.parent()
    n = parent.arity()
    base_ring = parent.base_ring()
    target = Associative(n, base_ring)
    result = target.zero()
    for key, coeff in element:
        assoc_dict = parent._assoc_from_basis_key(key)
        for word, word_coeff in assoc_dict.items():
            result += (coeff * word_coeff) * target(word)
    return result


lie_to_ass = OperadMorphism(Lie, Associative, _lie_to_ass_on_element)
"""PBW inclusion ``Lie → Ass``.

Sends a Lie bracket ``[x_i, ...]`` to the corresponding commutator
in the associative operad.
"""
