"""Canonical algebra/coalgebra model helpers."""

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
)
from uconf.algebraic.free_algebra import FreeAlgebraModule, FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.simplicial import (
    SurjectionSimplicialChainCoalgebra,
    SurjectionSimplicialCochainAlgebra,
    surjection_chain_action,
    surjection_cochain_action,
)
from uconf.algebraic.spherical import (
    ReducedSphereCochains,
    SurjectionSphereCochainAlgebra,
)
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.configuration import (
    labelled_configuration_model,
    unordered_configuration_model,
    euclidean_unordered_configuration_model,
)

__all__ = [
    "OperadAlgebra",
    "CooperadCoalgebra",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "HadamardTensorAlgebra",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSimplicialChainCoalgebra",
    "surjection_chain_action",
    "surjection_cochain_action",
    "ReducedSphereCochains",
    "SurjectionSphereCochainAlgebra",
    "PullbackAlgebra",
    "labelled_configuration_model",
    "unordered_configuration_model",
    "euclidean_unordered_configuration_model",
]
