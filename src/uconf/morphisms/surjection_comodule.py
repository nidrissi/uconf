"""Surjection-valued comodule morphism Δ': Ω(C) → S ⊙ Ω(C).

Postcomposes the E-comodule morphism of
:func:`uconf.morphisms.e_comodule_morphism.make_e_comodule_morphism`
with table reduction on the left factor.  This is the comodule morphism
used by the configuration models whose manifold model is a
``Surjection``-algebra (the Euclidean/sphere models of
:mod:`uconf.algebraic.configuration` and the simplicial torus model of
:mod:`uconf.algebraic.torus_simplicial`).
"""

from __future__ import annotations

from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.cooperad import CooperadLike
from uconf.core.morphism import OperadMorphism
from uconf.models import _compute_table_reduction_cached
from uconf.models.surjection import Surjection
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.wrappers.hadamard_operad import HadamardProduct


def make_surjection_comodule_morphism(cooperad_cls: CooperadLike) -> OperadMorphism:
    """Build the operad morphism Ω(C) → S ⊙ Ω(C).

    Composes the e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) (where
    E = BarrattEccles) with table reduction on the left factor, yielding
    a morphism into S ⊙ Ω(C) (where S = Surjection).
    """
    be_comodule = make_e_comodule_morphism(cooperad_cls)
    cobar = CobarConstruction(cooperad_cls)
    surj_target = HadamardProduct(Surjection, cobar)

    def _on_element(element):
        # Apply the BE-valued e-comodule morphism
        be_had_elem = be_comodule(element)

        # Apply table_reduction to the left (BarrattEccles) factor
        source_parent = be_had_elem.parent()
        n = source_parent.arity()
        base_ring = source_parent.base_ring()
        target_n = surj_target(n, base_ring)

        # Cache table_reduction results per BE basis key within this element,
        # avoiding redundant computation for repeated keys.
        tr_cache: dict = {}

        # Accumulate result as {key: coeff} dict to avoid repeated element
        # construction overhead.
        result_dict: dict = {}
        R = target_n.base_ring()

        be_parent = source_parent.left_parent()
        be_n = be_parent.arity()
        be_ring = be_parent.base_ring()
        for (be_key, cobar_key), coeff in be_had_elem:
            if be_key in tr_cache:
                surj_elem = tr_cache[be_key]
            else:
                # Call the cached table_reduction function directly,
                # bypassing the BE element construction + morphism overhead.
                surj_elem = _compute_table_reduction_cached(be_n, be_ring, be_key)
                tr_cache[be_key] = surj_elem

            for surj_key, surj_coeff in surj_elem:
                combined_key = (surj_key, cobar_key)
                combined_coeff = R(coeff * surj_coeff)
                if combined_key in result_dict:
                    result_dict[combined_key] += combined_coeff
                else:
                    result_dict[combined_key] = combined_coeff

        return target_n._from_dict(result_dict, remove_zeros=True)

    return OperadMorphism(cobar, surj_target, _on_element)
