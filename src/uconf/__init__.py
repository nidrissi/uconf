"""Utilities and operadic models for configuration-space computations.

This package exposes several operads and related constructions used in the
project:

- :class:`Surjection`
- :class:`SurjectionDual`
- :class:`BarrattEccles`
- :class:`Lie`
- :class:`Associative`
- :class:`Commutative`
- :class:`CooperadComponent`

It also wires two standard maps at import time:

- ``BarrattEccles.Element.table_reduction -> Surjection.Element``
- ``Surjection.Element.section -> BarrattEccles.Element``

These are implemented as lazy module morphisms cached on the parent objects.
"""

from uconf.algebraic import (
    CofreeCoalgebraModule,
    CofreeConilpotentCoalgebra,
    CooperadCoalgebra,
    FreeAlgebraModule,
    FreeOperadAlgebra,
    HadamardTensorAlgebra,
    OperadAlgebra,
    SurjectionSimplicialChainCoalgebra,
    SurjectionSimplicialCochainAlgebra,
    ReducedSphereCochains,
    SurjectionSphereCochainAlgebra,
    surjection_chain_action,
    surjection_cochain_action,
    labelled_configuration_model,
    unordered_configuration_model,
    euclidean_unordered_configuration_model,
)
from uconf.constructions import (
    BarConstruction,
    CobarConstruction,
    BarAlgebra,
    BarAlgebraModule,
    CobarCoalgebra,
    CobarCoalgebraModule,
)
from uconf.core import CooperadComponent, OperadComponent, OperadMorphism, TwistingMorphism
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.models import (
    Associative,
    BarrattEccles,
    CoAssociative,
    CoCommutative,
    Commutative,
    Lie,
    SimplicialChains,
    SimplicialCochains,
    Surjection,
    SurjectionDual,
)
from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator
from uconf.wrappers import HadamardProduct, ShiftedCooperad, ShiftedOperad
from uconf.homology import compute_chain_complex, homology_basis, compute_homology_representatives
from uconf.sampling import (
    random_surjection,
    random_surjection_key,
    random_planar_surjection,
    random_planar_surjection_key,
    random_sphere_admissible_surjection,
    random_sphere_admissible_surjection_key,
    random_lie_key,
    random_lie_element,
    random_barratt_eccles_key,
    random_barratt_eccles_element,
    random_hadamard_key,
    random_shuffle_tree,
    random_bar_element,
    random_cobar_element,
    random_free_algebra_element,
    random_cofree_coalgebra_element,
    random_tree_module_element,
    sample_basis,
    sample_operad_basis,
    sample_hadamard_basis,
    sample_algebra_pool,
    sphere_nontrivial_surjection_iter,
    sphere_nontrivial_operad_basis_iter,
)
from uconf.morphisms import (
    ass_to_com,
    lie_to_ass,
    make_e_comodule_morphism,
    canonical_projection,
    canonical_inclusion,
)
from uconf.tikz import (
    element_to_tikz,
    tree_to_forest,
    reps_to_tex_document,
    Layer,
)

__all__ = [
    "OperadComponent",
    "OperadMorphism",
    "TwistingMorphism",
    "PullbackAlgebra",
    "CooperadComponent",
    "Surjection",
    "SurjectionDual",
    "BarrattEccles",
    "Lie",
    "Associative",
    "Commutative",
    "CoAssociative",
    "CoCommutative",
    "SimplicialChains",
    "SimplicialCochains",
    "ShiftedOperad",
    "ShiftedCooperad",
    "HadamardProduct",
    "OperadAlgebra",
    "CooperadCoalgebra",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "HadamardTensorAlgebra",
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "BarConstruction",
    "CobarConstruction",
    "BarAlgebra",
    "BarAlgebraModule",
    "CobarCoalgebra",
    "CobarCoalgebraModule",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSimplicialChainCoalgebra",
    "ReducedSphereCochains",
    "SurjectionSphereCochainAlgebra",
    "surjection_chain_action",
    "surjection_cochain_action",
    "e_comodule_on_generator",
    "ass_to_com",
    "lie_to_ass",
    "make_e_comodule_morphism",
    "canonical_projection",
    "canonical_inclusion",
    "labelled_configuration_model",
    "unordered_configuration_model",
    "euclidean_unordered_configuration_model",
    "compute_chain_complex",
    "compute_homology_representatives",
    "homology_basis",
    "random_surjection",
    "random_surjection_key",
    "random_planar_surjection",
    "random_planar_surjection_key",
    "random_sphere_admissible_surjection",
    "random_sphere_admissible_surjection_key",
    "random_lie_key",
    "random_lie_element",
    "random_barratt_eccles_key",
    "random_barratt_eccles_element",
    "random_hadamard_key",
    "random_shuffle_tree",
    "random_bar_element",
    "random_cobar_element",
    "random_free_algebra_element",
    "random_cofree_coalgebra_element",
    "random_tree_module_element",
    "sample_basis",
    "sample_operad_basis",
    "sample_hadamard_basis",
    "sample_algebra_pool",
    "sphere_nontrivial_surjection_iter",
    "sphere_nontrivial_operad_basis_iter",
    "element_to_tikz",
    "tree_to_forest",
    "reps_to_tex_document",
    "Layer",
]
