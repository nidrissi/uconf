"""Canonical algebra/coalgebra model helpers."""

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
)
from uconf.algebraic.configuration import (
    euclidean_unordered_configuration_model,
    labelled_configuration_model,
    unordered_configuration_model,
)
from uconf.algebraic.free_algebra import FreeAlgebraModule, FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
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

__all__ = [
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "CooperadCoalgebra",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "HadamardTensorAlgebra",
    "OperadAlgebra",
    "PullbackAlgebra",
    "ReducedSphereCochains",
    "SurjectionSimplicialChainCoalgebra",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSphereCochainAlgebra",
    "euclidean_unordered_configuration_model",
    "labelled_configuration_model",
    "surjection_chain_action",
    "surjection_cochain_action",
    "unordered_configuration_model",
]
