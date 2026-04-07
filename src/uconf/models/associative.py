"""Associative operad model on permutation basis elements.

The component in arity ``n`` is the free module on ``S_n`` (for ``n >= 1``),
with zero differential.
"""

from __future__ import annotations

import itertools
from typing import ClassVar, Iterator

from sage.all import (
    CombinatorialFreeModule,
    Family,
    GradedModulesWithBasis,
    SymmetricGroup,
    SymmetricGroupAlgebra,
    cached_method,
    tensor,
)
from uconf.core.display import latex_linear_combination
from uconf.core.parented_element import ParentedElementMixin


class Associative(CombinatorialFreeModule):
    """Associative operad component in fixed arity."""

    name: ClassVar[str] = "Ass"
    connectivity: ClassVar[int] = 0
    """All components live in non-negative degrees."""

    def __init__(self, n: int, base_ring):
        """Initialize ``Ass(n)`` over ``base_ring``."""
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
        self.boundary = self.module_morphism(on_basis=lambda x: self.zero(), codomain=self)
        self.planarize = self.module_morphism(
            on_basis=self._planarize_on_basis,
            codomain=tensor([self, SymmetricGroupAlgebra(base_ring, n)]),
        )

    def _basis_keys(self) -> list[tuple[int, ...]]:
        if self.arity() == 0:
            return []
        return list(itertools.permutations(range(1, self.arity() + 1), self.arity()))

    def _validate_basis_key(self, basis_key: tuple | list) -> tuple[int, ...] | None:
        """Validate and normalize one basis key."""
        if self.arity() == 0:
            return None
        if not isinstance(basis_key, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple/list, got {type(basis_key)}")

        clean = tuple(int(i) for i in basis_key)
        n = self.arity()
        if len(clean) != n:
            raise ValueError(f"Basis key in arity {n} must have length {n}. Got {len(clean)}.")
        if set(clean) != set(range(1, n + 1)):
            raise ValueError(f"Basis key must be a permutation of 1..{n}. Got {clean}.")
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

        raise TypeError(f"Expected dict or tuple/list; got {type(x).__name__}: {x!r}.")

    def arity(self) -> int:
        """Return the fixed arity of this operad component."""
        return self._arity

    @staticmethod
    def unit(base_ring):
        """Return the operadic unit in arity ``1``."""
        return Associative(1, base_ring)((1,))

    @staticmethod
    def unit_key() -> tuple:
        """Return the basis key of the unit element in arity ``1``."""
        return (1,)

    def basis_iter(self, d: int) -> Iterator[Element]:
        """Iterate over basis elements in this arity and the given degree."""
        if d == 0:
            for key in self._basis_keys():
                yield self(key)

    def planar_basis_iter(self, d: int) -> Iterator[Element]:
        """Iterate over planar basis elements in this arity and the given degree."""
        if d == 0:
            yield self(tuple(range(1, self.arity() + 1)))

    @cached_method
    def graded_basis(self, d: int) -> Family:
        """Return the ``Family`` of all basis elements in degree ``d``."""
        return Family(self.basis_iter(d))

    @cached_method
    def graded_planar_basis(self, d: int) -> Family:
        """Return the ``Family`` of planar basis elements in degree ``d``."""
        return Family(self.planar_basis_iter(d))

    def _planarize_on_basis(self, basis_element: tuple):
        """Split into planar representative and symmetric-group factor."""
        n = self.arity()
        sigma = self._symmetric_group(list(basis_element))
        planar = tuple(range(1, n + 1))
        return self.term(planar).tensor(SymmetricGroupAlgebra(self.base_ring(), n)(sigma))

    def degree_on_basis(self, basis_element: tuple) -> int:
        """Return homological degree of one basis element."""
        return 0

    def _repr_term(self, basis_element: tuple) -> str:
        return f"μ({basis_element})"

    def _latex_term(self, basis_element: tuple) -> str:
        entries = ",".join(str(i) for i in basis_element)
        return f"\\mu_{{{entries}}}"

    @staticmethod
    def _compose_basis_tuple(
        sigma: tuple[int, ...], i: int, tau: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Compose two permutation basis tuples at input ``i``."""
        shift = len(tau) - 1
        result = []
        for value in sigma:
            if value < i:
                result.append(value)
            elif value > i:
                result.append(value + shift)
            else:
                result.extend([t + i - 1 for t in tau])
        return tuple(result)

    @staticmethod
    def compose(
        x: Element,
        input: int,
        y: Element,
    ):
        """Operadic composition ``x \\circ_i y``."""
        if x.parent().base_ring() != y.parent().base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x.arity()
        n = y.arity()
        assert 1 <= input <= m, f"Index i must be between 1 and {m}. Got {input}."

        target = Associative(m + n - 1, base_ring=x.parent().base_ring())

        def term_generator():
            for sigma, x_coeff in x:
                for tau, y_coeff in y:
                    yield (
                        Associative._compose_basis_tuple(sigma, input, tau),
                        target.base_ring()(x_coeff * y_coeff),
                    )

        return target.sum_of_terms(term_generator())

    class Element(ParentedElementMixin["Associative"], CombinatorialFreeModule.Element):
        """Elements of a fixed-arity associative component."""

        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))

        def arity(self) -> int:
            """Return the arity of this element."""
            return self.parent().arity()

        def boundary(self):
            """Apply the differential."""
            parent = self.parent()
            return parent.boundary(self)

        def planarize(self):
            """Project to planar representative tensored with a group element."""
            parent = self.parent()
            return parent.planarize(self)

        def permute(self, sigma):
            """Permute labels in each supported basis permutation."""
            parent = self.parent()

            if isinstance(sigma, (list, tuple)):
                sigma = parent._symmetric_group(sigma)
            elif not (hasattr(sigma, "parent") and sigma.parent() == parent._symmetric_group):
                raise TypeError(
                    f"Permutation must be a list, tuple, or S_{parent.arity()} element; "
                    f"got {type(sigma).__name__}: {sigma!r}."
                )

            def term_generator():
                R = parent.base_ring()
                for basis_key, coeff in self:
                    permuted = tuple(sigma(v) for v in basis_key)
                    yield permuted, R(coeff)

            return parent.sum_of_terms(term_generator())
