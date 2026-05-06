"""Canonical operad/cooperad and simplicial model classes."""

from sage.all import cached_function

from itertools import combinations

from uconf.models.associative import Associative
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.coassociative import CoAssociative
from uconf.models.cocommutative import CoCommutative
from uconf.models.commutative import Commutative
from uconf.models.lie import Lie
from uconf.models.simplicial import SimplicialChains, SimplicialCochains
from uconf.models.surjection import Surjection
from uconf.models.surjection_dual import SurjectionDual

__all__ = [
    "Associative",
    "BarrattEccles",
    "CoAssociative",
    "CoCommutative",
    "Commutative",
    "Lie",
    "SimplicialChains",
    "SimplicialCochains",
    "Surjection",
    "SurjectionDual",
]


def _compositions(m: int, k: int):
    """Yield all compositions of m into k positive parts.

    Equivalent to Partitions(m, length=k) + all permutations, but uses a
    direct combinatorial construction (choose k-1 dividers from m-1 slots)
    that avoids Sage type overhead and the permutation-dedup set.
    """
    for dividers in combinations(range(1, m), k - 1):
        prev = 0
        parts: list[int] = []
        for d in dividers:
            parts.append(d - prev)
            prev = d
        parts.append(m - prev)
        yield tuple(parts)


@cached_function
def _compute_table_reduction_cached(n, base_ring, basis_element: tuple) -> Surjection.Element:
    """Compute table reduction on one Barratt--Eccles basis element (cached).

    Cached on (arity, base_ring, basis_element) so repeated calls with the
    same BE key avoid recomputing the expensive composition enumeration.
    """
    d = len(basis_element) - 1
    target = Surjection(n, base_ring)
    # Hoist .tuple() calls: each permutation object is queried once here
    # instead of once per composition inside the generator.
    basis_tuples = tuple(s.tuple() for s in basis_element)
    R_one = base_ring.one()

    def term_generator():
        for pi in _compositions(d + n, d + 1):
            k2: list[int] = []
            removed_mask = 0  # bitmask: bit v set <=> v is in "removed"
            degenerate = False
            for idx, i in enumerate(pi):
                filtered = [v for v in basis_tuples[idx] if not (removed_mask & (1 << v))]
                if idx > 0 and k2[-1] == filtered[0]:
                    degenerate = True
                    break
                if i > 1:
                    for v in filtered[: i - 1]:
                        removed_mask |= 1 << v
                k2 += filtered[:i]
            if not degenerate:
                yield tuple(k2), R_one

    return target.sum_of_terms(term_generator())


def _table_reduction_on_basis(self: BarrattEccles):
    """Create the basis-level table-reduction map ``E_n -> S_n``.

    Returns a callable compatible with ``module_morphism(on_basis=...)``.
    """
    n = self.arity()
    base_ring = self.base_ring()

    def _compute_table_reduction(basis_element: tuple) -> Surjection.Element:
        return _compute_table_reduction_cached(n, base_ring, basis_element)

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


BarrattEccles.Element.table_reduction = _table_reduction


Surjection.Element.section = _section
