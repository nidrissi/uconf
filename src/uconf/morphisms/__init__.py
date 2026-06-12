"""Operad morphisms, pullback algebras, and classical maps."""

from uconf.morphisms.classical import ass_to_com, lie_to_ass
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.morphisms.surjection_comodule import make_surjection_comodule_morphism
from uconf.morphisms.canonical_twisting import canonical_projection, canonical_inclusion

__all__ = [
    "ass_to_com",
    "lie_to_ass",
    "make_e_comodule_morphism",
    "make_surjection_comodule_morphism",
    "canonical_projection",
    "canonical_inclusion",
]
