from __future__ import annotations
from typing import Any, Protocol, Self, runtime_checkable, Iterator


@runtime_checkable
class ComponentProtocol(Protocol):
    """Structural contract for a component of an operad/cooperad ."""

    name: str
    """Name of the component, used for printing."""

    connectivity: int
    """Lower bound k such that M(n) is concentrated in degrees >= k*(n-1) for all n >= 0."""

    def __init__(self, n: int, base_ring):
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def base_ring(self) -> Any:
        """Returns the base ring of this component."""
        ...

    def __call__(self, input) -> Element:
        """Returns the basis element corresponding to basis key ``input``."""
        ...

    def _validate_basis_key(self, x: object) -> Any:
        """Validates and normalizes a basis key (implementation detail)."""
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

    def degree_on_basis(self, x: object) -> int:
        """Returns degree of basis element ``x``."""
        ...

    def sum_of_terms(self, terms: Iterator[tuple[Any, Any]]) -> Element:
        """Builds element from iterator of (basis, coeff) pairs."""
        ...

    def zero(self) -> Element:
        """Returns the zero element of this component."""
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

        # This is ugly but I cannot find a better way to express the fact that the element
        # class should inherit from CombinatorialFreeModule.Element.
        def __iter__(self) -> Iterator[tuple[Any, Any]]: ...
        def __rmul__(self, other) -> Self: ...
        def parent(self) -> Any: ...
