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
    ReducedSphereCochains,
    SurjectionSimplicialChainCoalgebra,
    SurjectionSimplicialCochainAlgebra,
    SurjectionSphereCochainAlgebra,
    euclidean_unordered_configuration_model,
    labelled_configuration_model,
    surjection_chain_action,
    surjection_cochain_action,
    unordered_configuration_model,
)
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.constructions import (
    BarAlgebra,
    BarAlgebraModule,
    BarConstruction,
    CobarCoalgebra,
    CobarCoalgebraModule,
    CobarConstruction,
)
from uconf.core import CooperadComponent, OperadComponent, OperadMorphism, TwistingMorphism
from uconf.homology import compute_chain_complex, compute_homology_representatives, homology_basis
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
from uconf.morphisms import (
    ass_to_com,
    canonical_inclusion,
    canonical_projection,
    lie_to_ass,
    make_e_comodule_morphism,
)
from uconf.morphisms.e_comodule_morphism import e_comodule_on_generator
from uconf.sampling import (
    random_bar_element,
    random_barratt_eccles_element,
    random_barratt_eccles_key,
    random_cobar_element,
    random_cofree_coalgebra_element,
    random_free_algebra_element,
    random_hadamard_key,
    random_lie_element,
    random_lie_key,
    random_planar_surjection,
    random_planar_surjection_key,
    random_shuffle_tree,
    random_sphere_admissible_surjection,
    random_sphere_admissible_surjection_key,
    random_surjection,
    random_surjection_key,
    random_tree_module_element,
    sample_algebra_pool,
    sample_basis,
    sample_hadamard_basis,
    sample_operad_basis,
    sphere_nontrivial_operad_basis_iter,
    sphere_nontrivial_surjection_iter,
)
from uconf.tikz import (
    Layer,
    element_to_tikz,
    reps_to_tex_document,
    tree_to_forest,
)
from uconf.wrappers import HadamardProduct, ShiftedCooperad, ShiftedOperad

__all__ = [
    "Associative",
    "BarAlgebra",
    "BarAlgebraModule",
    "BarConstruction",
    "BarrattEccles",
    "CoAssociative",
    "CoCommutative",
    "CobarCoalgebra",
    "CobarCoalgebraModule",
    "CobarConstruction",
    "CofreeCoalgebraModule",
    "CofreeConilpotentCoalgebra",
    "Commutative",
    "CooperadCoalgebra",
    "CooperadComponent",
    "FreeAlgebraModule",
    "FreeOperadAlgebra",
    "HadamardProduct",
    "HadamardTensorAlgebra",
    "Layer",
    "Lie",
    "OperadAlgebra",
    "OperadComponent",
    "OperadMorphism",
    "PullbackAlgebra",
    "ReducedSphereCochains",
    "ShiftedCooperad",
    "ShiftedOperad",
    "SimplicialChains",
    "SimplicialCochains",
    "Surjection",
    "SurjectionDual",
    "SurjectionSimplicialChainCoalgebra",
    "SurjectionSimplicialCochainAlgebra",
    "SurjectionSphereCochainAlgebra",
    "TwistingMorphism",
    "ass_to_com",
    "canonical_inclusion",
    "canonical_projection",
    "compute_chain_complex",
    "compute_homology_representatives",
    "e_comodule_on_generator",
    "element_to_tikz",
    "euclidean_unordered_configuration_model",
    "homology_basis",
    "labelled_configuration_model",
    "lie_to_ass",
    "make_e_comodule_morphism",
    "random_bar_element",
    "random_barratt_eccles_element",
    "random_barratt_eccles_key",
    "random_cobar_element",
    "random_cofree_coalgebra_element",
    "random_free_algebra_element",
    "random_hadamard_key",
    "random_lie_element",
    "random_lie_key",
    "random_planar_surjection",
    "random_planar_surjection_key",
    "random_shuffle_tree",
    "random_sphere_admissible_surjection",
    "random_sphere_admissible_surjection_key",
    "random_surjection",
    "random_surjection_key",
    "random_tree_module_element",
    "reps_to_tex_document",
    "sample_algebra_pool",
    "sample_basis",
    "sample_hadamard_basis",
    "sample_operad_basis",
    "sphere_nontrivial_operad_basis_iter",
    "sphere_nontrivial_surjection_iter",
    "surjection_chain_action",
    "surjection_cochain_action",
    "tree_to_forest",
    "unordered_configuration_model",
]
