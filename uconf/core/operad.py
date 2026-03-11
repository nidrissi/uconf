"""Typing protocols for operad-like objects used in this project."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, Self, TypeAlias, runtime_checkable


@runtime_checkable
class OperadProtocol(Protocol):
    """Structural contract for one fixed-arity operad component."""

    name: str
    """Name of the operad, used for printing."""

    def __init__(self, n: int, base_ring=...):
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def __call__(self, input) -> Element:
        """Returns the basis element corresponding to basis key ``input``."""
        ...

    def arity(self) -> int:
        """Returns the arity of this operad component."""
        ...

    def term(self, x: object) -> Element:
        """Returns the basis element corresponding to basis key ``x``."""
        ...

    def boundary(self, x: Element) -> Element:
        """Applies the differential to element ``x``."""
        ...

    @staticmethod
    def compose(x: Element, input: int, y: Element) -> Element:
        """Computes ``x \\circ_i y``."""
        ...

    def degree_on_basis(self, x: object) -> int:
        """Returns degree of basis element ``x``."""
        ...

    @staticmethod
    def unit() -> Element:
        """Returns the unit element of the operad."""
        ...

    def base_ring(self) -> Any:
        """Returns the base ring of this component."""
        ...

    def sum_of_terms(self, terms: Iterator[tuple[Any, Any]]) -> Element:
        """Builds element from iterator of (basis, coeff) pairs."""
        ...

    def _validate_basis_key(self, x: object) -> Any:
        """Validates and normalizes a basis key (implementation detail)."""
        ...

    class Element(Protocol):
        """Structural contract for elements of an operad component.

        Elements represent sparse linear combinations and are iterable over
        (basis_key, coefficient) pairs.
        """

        def arity(self) -> int:
            """Returns the arity of this element."""
            ...

        def boundary(self) -> Self:
            """Returns the boundary of this element."""
            ...

        def permute(self, sigma: object) -> Self:
            """Returns the result of permuting this element by ``sigma``."""
            ...

        # This is ugly but I cannot find a better way to express the fact that the element class should inherit from CombinatorialFreeModule.Element.
        def __iter__(self) -> Iterator[tuple[Any, Any]]: ...
        def __rmul__(self, other) -> Self: ...


@runtime_checkable
class OperadFactoryProtocol(Protocol):
    """Structural contract for operad factories and wrappers.

    Factory instances (for example shifted/Hadamard wrappers) are callable by
    arity and expose operadic structure maps at the factory level.
    """

    name: str

    def __call__(self, n: int, base_ring=...) -> Any:
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def unit(self, base_ring=...) -> Any:
        """Returns the unit element of the operad."""
        ...

    def compose(
        self,
        x: Any,
        i: int,
        y: Any,
    ) -> Any:
        """Computes ``x \\circ_i y``."""
        ...


OperadLike: TypeAlias = type[OperadProtocol] | OperadFactoryProtocol
"""Type alias for accepted operad inputs (class or factory instance)."""
