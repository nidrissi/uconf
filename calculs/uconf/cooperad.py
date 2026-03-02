"""Typing protocol for cooperad-like objects used in this project."""

from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable


C = TypeVar("C", bound="CooperadProtocol")


@runtime_checkable
class CooperadProtocol(Protocol[C]):
    """Structural contract for cooperad implementations.

    This protocol mirrors :mod:`uconf.operad` and documents the minimal dual
    operations expected by bar/cobar constructions.
    """

    name: ClassVar[str]
    """Name of the cooperad, used for printing."""

    def __init__(self, n, base_ring):
        """Initializes the cooperad in arity n over the given base ring."""
        ...

    def arity(self) -> int:
        """Returns the arity of this cooperad component."""
        ...

    def degree_on_basis(self, x: C) -> int:
        """Returns degree of element x."""
        ...

    @staticmethod
    def counit(x: C):
        """Returns the counit evaluation on x."""
        ...

    @staticmethod
    def reduced(x: C) -> C:
        """Projects x to the reduced part (kills counit in arity 1)."""
        ...

    @staticmethod
    def infinitesimal_cocompose(x: C, i: int, m: int, n: int):
        """Returns the partial cocomposition in slot i, arities (m, n)."""
        ...

    class Element:
        """Protocol for elements living in a cooperad object."""

        def arity(self) -> int:
            """Returns the arity of this element."""
            ...

        def boundary(self) -> C:
            """Returns the boundary of this element."""
            ...

        def permute(self, sigma: Any) -> C:
            """Returns the result of permuting this element by sigma."""
            ...

        def counit(self):
            """Returns the counit evaluation of this element."""
            ...

        def reduced(self) -> C:
            """Returns the reduced projection of this element."""
            ...

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            """Returns the partial cocomposition of this element."""
            ...
