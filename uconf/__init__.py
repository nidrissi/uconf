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

from itertools import permutations

from sage.all import Partitions

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
    BarComplexAlgebra,
    BarConstruction,
    CobarComplexCoalgebra,
    CobarConstruction,
    TwistedBarComplexAlgebra,
    TwistedBarComplex,
    TwistedCobarComplex,
    e_comodule_on_generator,
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
from uconf.wrappers import HadamardProduct, ShiftedCooperad, ShiftedOperad
from uconf.homology import chain_complex, homology_basis
from uconf.morphisms import ass_to_com, lie_to_ass, make_e_comodule_morphism, canonical_projection, canonical_inclusion

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
    "BarComplexAlgebra",
    "CobarComplexCoalgebra",
    "TwistedBarComplexAlgebra",
    "TwistedBarComplex",
    "TwistedCobarComplex",
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
    "chain_complex",
    "homology_basis",
]


def _table_reduction_on_basis(self: BarrattEccles):
    """Create the basis-level table-reduction map ``E_n -> S_n``.

    Returns a callable compatible with ``module_morphism(on_basis=...)``.
    """

    def _compute_table_reduction(basis_element: tuple) -> Surjection.Element:
        """Compute table reduction on one Barratt--Eccles basis element."""
        n = self.arity()
        d = len(basis_element) - 1
        target = Surjection(n, self.base_ring())

        def term_generator():
            for pi_ord in Partitions(
                d + n,
                length=d + 1,  # pyright: ignore[reportCallIssue]
            ):
                for pi in set(permutations(pi_ord)):
                    k2, removed = [], []
                    degenerate = False
                    for idx, i in enumerate(pi):
                        filtered = [i for i in basis_element[idx].tuple() if i not in removed]
                        if idx > 0 and k2[-1] == filtered[0]:
                            degenerate = True
                            break
                        if i > 1:
                            removed += filtered[: i - 1]
                        k2 += filtered[:i]
                    if not degenerate:
                        yield tuple(k2), 1

        return target.sum_of_terms(term_generator())

    return _compute_table_reduction


def _table_reduction(self: BarrattEccles.Element) -> Surjection.Element:
    """Evaluate table reduction on an element of the Barratt--Eccles operad."""
    parent = self.parent()
    if not hasattr(parent, "_table_reduction"):
        parent._table_reduction = parent.module_morphism(
            on_basis=_table_reduction_on_basis(parent),
            codomain=Surjection(parent.arity(), parent.base_ring()),
        )
    return parent._table_reduction(self)


BarrattEccles.Element.table_reduction = _table_reduction


def _section_on_basis(self: Surjection):
    """Create the basis-level section map ``S_n -> E_n``.

    Returns a callable compatible with ``module_morphism(on_basis=...)``.
    """

    def _compute_section(u: tuple) -> BarrattEccles.Element:
        """Compute the section image of one surjection basis element."""
        n = self.arity()
        target = BarrattEccles(n, self.base_ring())
        caesura_indices = Surjection._caesuras(u)
        sections = [[i for i in range(len(u)) if i not in caesura_indices]]
        for d in reversed(caesura_indices):
            # add a new element to section based on the last element
            # drop the element that maps to u[d]
            # add d to the new element
            new_section = sections[-1][:]
            to_remove = None
            for i, v in enumerate(new_section):
                if u[v] == u[d]:
                    to_remove = i
                    break
            assert to_remove is not None
            new_section.pop(to_remove)
            new_section.append(d)
            new_section.sort()
            sections.append(new_section)
        sections.reverse()
        return target(tuple([u[i] for i in section] for section in sections))

    return _compute_section


def _section(self: Surjection.Element) -> BarrattEccles.Element:
    """Evaluate the section map on an element of the surjection operad."""
    parent = self.parent()
    if not hasattr(parent, "_section"):
        parent._section = parent.module_morphism(
            on_basis=_section_on_basis(parent),
            codomain=BarrattEccles(parent.arity(), parent.base_ring()),
        )
    return parent._section(self)


Surjection.Element.section = _section
