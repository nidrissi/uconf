"""E-comodule morphism Δ: Ω(C) → E ⊗ Ω(C).

For a quasi-planar cooperad C, the E-comodule map defines an operad morphism
from the cobar construction Ω(C) to the Hadamard product E ⊗ Ω(C), where
E = BarrattEccles is the Barratt–Eccles operad.

Since Ω(C) is the free operad T(s⁻¹C̄), the morphism is uniquely determined
by its values on generators (single-vertex trees).  On generators, the map is
computed by :func:`~uconf.constructions.e_comodule.e_comodule_on_generator`.
The extension to arbitrary trees uses the universal property of the free operad.
"""

from __future__ import annotations

from typing import Any

from uconf.constructions.cobar_construction import CobarConstruction
from uconf.constructions.e_comodule import e_comodule_on_generator
from uconf.core.morphism import OperadMorphism
from uconf.core.trees import children, decoration, is_leaf, tree_arity, vertex_arity
from uconf.models.barratt_eccles import BarrattEccles
from uconf.wrappers.hadamard_operad import HadamardProduct


def make_e_comodule_morphism(
    cooperad_cls: Any,
) -> OperadMorphism:
    """Build the operad morphism Δ: Ω(C) → E ⊗ Ω(C).

    Parameters
    ----------
    cooperad_cls :
        A quasi-planar cooperad (class or factory) for which the cobar
        construction Ω(C) is defined.

    Returns
    -------
    OperadMorphism
        The morphism Δ whose source is ``CobarConstruction(cooperad_cls)``
        and whose target is ``HadamardProduct(BarrattEccles, CobarConstruction(cooperad_cls))``.
    """
    cobar = CobarConstruction(cooperad_cls)
    target_factory = HadamardProduct(BarrattEccles, cobar)

    def _extend_tree(tree: Any, base_ring: Any) -> Any:
        """Extend the morphism to a single tree by the free operad universal property."""
        if is_leaf(tree):
            return target_factory.unit(base_ring)

        dec = decoration(tree)
        kids = children(tree)
        k = vertex_arity(tree)

        # Map the root generator via e_comodule_on_generator
        cooperad_parent = cooperad_cls(k, base_ring)
        gen_elem = cooperad_parent.term(dec)
        root_tensor = e_comodule_on_generator(gen_elem)

        # Convert the tensor([BE(k), Ω(C)(k)]) element to HadamardProduct element
        target_k = target_factory(k, base_ring)
        root_image = target_k.zero()
        for tensor_basis, coeff in root_tensor:
            be_key, cobar_key = tensor_basis
            root_image += coeff * target_k.term((be_key, cobar_key))

        # Compose with child images from right to left (∘_k, ∘_{k-1}, ..., ∘_1)
        # This preserves input positions 1, ..., j-1 at each step.
        result = root_image
        for j in range(k, 0, -1):
            child = kids[j - 1]
            if is_leaf(child):
                # Composing with the unit is the identity, skip
                continue
            child_image = _extend_tree(child, base_ring)
            result = target_factory.compose(result, j, child_image)

        return result

    def _on_element(element: Any) -> Any:
        """Apply the morphism to an arbitrary cobar element by linearity."""
        parent = element.parent()
        n = parent.arity()
        base_ring = parent.base_ring()
        target_n = target_factory(n, base_ring)
        result = target_n.zero()
        for tree, coeff in element:
            result += coeff * _extend_tree(tree, base_ring)
        return result

    return OperadMorphism(cobar, target_factory, _on_element)
