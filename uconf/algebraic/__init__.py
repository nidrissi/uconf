"""Canonical algebra/coalgebra model helpers."""

from uconf.algebraic.algebra import OperadAlgebra
from uconf.algebraic.coalgebra import CooperadCoalgebra
from uconf.algebraic.cofree_coalgebra import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
)
from uconf.algebraic.free_algebra import FreeAlgebraModule, FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import (
    HadamardTensorAlgebra,
    hadamard_tensor_algebra,
)
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
    "OperadAlgebra",
    "CooperadCoalgebra",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "HadamardTensorAlgebra",
    "hadamard_tensor_algebra",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSimplicialChainCoalgebra",
    "surjection_chain_action",
    "surjection_cochain_action",
    "ReducedSphereCochains",
    "SurjectionSphereCochainAlgebra",
]
