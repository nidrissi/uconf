"""Commutative operad model.

For ``n >= 1``, ``Com(n)`` is rank-one with trivial symmetric action and zero
differential. Arity ``0`` is the zero module.
"""

from __future__ import annotations

from typing import ClassVar, Iterator

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, SymmetricGroup
from uconf.core.parented_element import ParentedElementMixin


class Commutative(CombinatorialFreeModule):
    """Commutative operad component in fixed arity."""

    name: ClassVar[str] = "Com"
    connectivity: ClassVar[int] = 0
    """All components live in non-negative degrees."""

    def __init__(self, n: int, base_ring):
        """Initialize ``Com(n)`` over ``base_ring``."""

        assert n >= 0, f"Arity must be non-negative. Got {n}."
        name = f"{self.name}{n}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._arity = int(n)
        self._symmetric_group = SymmetricGroup(n)
        self.boundary = self.module_morphism(
            on_basis=lambda basis: self.zero(), codomain=self
        )

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

    def _element_constructor_(self, x):
        """Build elements from basis keys or sparse dictionaries."""

        if isinstance(x, dict):
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is None:
                    continue
                clean_dict[clean_key] = coeff
            return super()._element_constructor_(clean_dict)

        if isinstance(x, (tuple, list)):
            clean_key = self._validate_basis_key(x)
            if clean_key is None:
                return self.zero()
            return self.term(clean_key)

        raise TypeError(
            f"Input must be a dictionary (for linear combinations) or a tuple/list (for basis elements). Got {x} ({type(x)})."
        )

    def arity(self) -> int:
        """Return the fixed arity of this operad component."""

        return self._arity

    @staticmethod
    def unit(base_ring):
        """Return the operadic unit in arity ``1``."""

        return Commutative(1, base_ring)(())

    def basis_it(self, d: int) -> Iterator[Commutative.Element]:
        """Iterate over basis elements in this arity and the given degree."""

        if self.arity() >= 1 and d == 0:
            yield self.term(())

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Return homological degree of one basis element."""

        return 0

    @staticmethod
    def compose(
        x,
        input: int,
        y,
    ):
        """Operadic composition ``x \\circ_i y``."""

        if x.parent().base_ring() != y.parent().base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x.arity()
        n = y.arity()
        assert 1 <= input <= m, f"Index i must be between 1 and {m}. Got {input}."

        target = Commutative(m + n - 1, base_ring=x.parent().base_ring())
        result = target.zero()
        for _, x_coeff in x:
            for _, y_coeff in y:
                result += target.term(()) * (x_coeff * y_coeff)
        return result

    class Element(ParentedElementMixin["Commutative"], CombinatorialFreeModule.Element):
        """Elements of a fixed-arity commutative component."""

        def arity(self) -> int:
            """Return the arity of this element."""

            return self._parent().arity()

        def boundary(self):
            """Apply the differential."""

            parent = self._parent()
            return parent.boundary(self)

        def permute(self, sigma):
            """Return the trivial symmetric-group action on this element."""

            parent = self._parent()

            if isinstance(sigma, (list, tuple)):
                parent._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{parent.arity()}. Got {sigma} ({type(sigma)})."
                )
            return self
