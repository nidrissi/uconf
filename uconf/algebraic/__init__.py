"""Canonical algebra/coalgebra model helpers."""

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
)
from uconf.algebraic.free_algebra import FreeAlgebraModule, FreeOperadAlgebra
from uconf.algebraic.simplicial import (
    SurjectionSimplicialChainCoalgebra,
    SurjectionSimplicialCochainAlgebra,
    surjection_chain_action,
    surjection_cochain_action,
)

__all__ = [
    "OperadAlgebra",
    "CooperadCoalgebra",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSimplicialChainCoalgebra",
    "surjection_chain_action",
    "surjection_cochain_action",
]
