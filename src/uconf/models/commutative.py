"""Commutative operad model.

For ``n >= 1``, ``Com(n)`` is rank-one with trivial symmetric action and zero
differential. Arity ``0`` is the zero module.
"""

from __future__ import annotations

from collections.abc import Iterator

from uconf.core.component import ComponentBase, ElementBase, SymmetricSequence
from uconf.core.operad import Operad


class CommutativeElement(ElementBase["CommutativeComponent"]):
    """Elements of a fixed-arity commutative component."""

    def permute(self, sigma) -> CommutativeElement:
        """Return the trivial symmetric-group action on this element."""
        self.parent()._coerce_permutation(sigma)
        return self


class CommutativeComponent(ComponentBase["CommutativeElement"]):
    """Arity-``n`` component: rank one for ``n >= 1``, zero in arity ``0``."""

    Element = CommutativeElement

    def __init__(self, factory: SymmetricSequence, n: int, base_ring):
        """Initialize ``Com(n)`` over ``base_ring``."""
        super().__init__(factory, n, base_ring)
        self.boundary = self.module_morphism(on_basis=lambda basis: self.zero(), codomain=self)

    def _validate_basis_key(self, basis_key: tuple | list) -> tuple | None:
        """Validate and normalize one basis key."""
        if self.arity() == 0:
            return None
        if not isinstance(basis_key, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple/list, got {type(basis_key)}")

        clean = tuple(basis_key)
        if clean != ():
            raise ValueError(f"The only basis key in arity {self.arity()} is ().")
        return clean

    def basis_iter(self, d: int) -> Iterator[CommutativeElement]:
        """Iterate over basis elements in this arity and the given degree."""
        if self.arity() >= 1 and d == 0:
            yield self.term(())

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Return homological degree of one basis element."""
        return 0

    def _repr_term(self, basis_element: tuple) -> str:
        return f"μ([{self.arity()}])"

    def _latex_term(self, basis_element: tuple) -> str:
        return "\\mu_{" + str(self.arity()) + "}"


class CommutativeOperad(Operad[CommutativeComponent]):
    """The commutative operad ``Com``; use the instance :data:`Commutative`."""

    Component = CommutativeComponent
    Element = CommutativeElement

    name = "Com"
    connectivity = 0
    """All components live in non-negative degrees."""

    def unit(self, base_ring) -> CommutativeElement:
        """Return the operadic unit in arity ``1``."""
        return self(1, base_ring)(())

    def unit_key(self) -> tuple:
        """Return the basis key of the unit element in arity ``1``."""
        return ()

    def compose(
        self,
        x: CommutativeElement,
        input: int,
        y: CommutativeElement,
    ) -> CommutativeElement:
        """Operadic composition ``x \\circ_i y``."""
        if x.parent().base_ring() != y.parent().base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x.arity()
        n = y.arity()
        assert 1 <= input <= m, f"Index i must be between 1 and {m}. Got {input}."

        target = self(m + n - 1, base_ring=x.parent().base_ring())
        result = target.zero()
        for _, x_coeff in x:
            for _, y_coeff in y:
                result += target.term(()) * (x_coeff * y_coeff)
        return result


Commutative = CommutativeOperad()
