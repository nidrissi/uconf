"""Typing protocols for cooperad-like objects used in this project."""

from collections.abc import Iterator
from typing import Any, Iterable, Protocol, Self, runtime_checkable


@runtime_checkable
class CooperadProtocol(Protocol):
    """Structural contract for one fixed-arity cooperad component."""

    name: str
    """Name of the cooperad, used for printing."""

    def __init__(self, n: int, base_ring=...):
        """Returns the arity-``n`` component over ``base_ring``."""
        ...

    def arity(self) -> int:
        """Returns the arity of this cooperad component."""
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

    @staticmethod
    def counit(x: Element) -> object:
        """Returns the counit evaluation on ``x``."""
        ...

    @staticmethod
    def reduced(x: Element) -> Element:
        """Projects ``x`` to the reduced part (kills counit in arity 1)."""
        ...

    @staticmethod
    def infinitesimal_cocompose(x: Element, i: int, m: int, n: int) -> Iterable[tuple]:
        """Returns the partial cocomposition in slot ``i``, arities ``(m, n)``."""
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
        """Structural contract for elements of a cooperad component.


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

        def counit(self) -> object:
            """Returns the counit evaluation of this element."""
            ...

        def reduced(self) -> Self:
            """Returns the reduced projection of this element."""
            ...

        def infinitesimal_cocompose(self, i: int, m: int, n: int) -> Iterable[tuple]:
            """Returns the partial cocomposition of this element."""
            ...

        # This is ugly but I cannot find a better way to express the fact that the element class should inherit from CombinatorialFreeModule.Element.
        def __iter__(self) -> Iterator[tuple[Any, Any]]: ...
        def __rmul__(self, other) -> Self: ...
