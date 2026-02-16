from .bar import *
from .barratt_eccles import *
from .operad import *
from .surjection import *


def _table_reduction_on_basis(self: BarrattEccles):
    def _compute_table_reduction(basis_element: tuple) -> Surjection.Element:
        n = self.arity()
        d = len(basis_element) - 1
        target = Surjection(n)

        def term_generator():
            for pi_ord in Partitions(
                d + n, length=d + 1  # pyright: ignore[reportCallIssue]
            ):
                for pi in set(permutations(pi_ord)):
                    k2, removed = [], []
                    degenerate = False
                    for idx, i in enumerate(pi):
                        filtered = [
                            i for i in basis_element[idx].tuple() if i not in removed
                        ]
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
    parent = self.parent()
    if not hasattr(parent, "_table_reduction"):
        parent._table_reduction = parent.module_morphism(
            on_basis=_table_reduction_on_basis(parent),
            codomain=Surjection(parent.arity()),
        )
    return parent._table_reduction(self)


BarrattEccles.Element.table_reduction = _table_reduction


def _section_on_basis(self: Surjection):
    def _compute_section(u: tuple) -> BarrattEccles.Element:
        n = self.arity()
        target = BarrattEccles(n)
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
    parent = self.parent()
    if not hasattr(parent, "_section"):
        parent._section = parent.module_morphism(
            on_basis=_section_on_basis(parent),
            codomain=BarrattEccles(parent.arity()),
        )
    return parent._section(self)


Surjection.Element.section = _section
