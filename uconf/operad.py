from typing import Protocol, Any, TypeVar, runtime_checkable


P = TypeVar("P", bound="OperadProtocol")


@runtime_checkable
class OperadProtocol(Protocol[P]):
    """
    Formal definition of what a Python object must do to be an 'Operad'.
    """

    name: str
    """Name of the operad, used for printing."""

    def arity(self) -> int:
        """Returns the arity of this operad."""
        ...

    @staticmethod
    def compose(x: P, i: int, y: P) -> P:
        """Computes x o_i y."""
        ...

    def degree_on_basis(self, x: P) -> int:
        """Returns degree of element x."""
        ...

    @staticmethod
    def unit() -> P:
        """Returns the unit element of the operad."""
        ...

    class Element:
        def arity(self) -> int:
            """Returns the arity of this element."""
            ...

        def boundary(self) -> P:
            """Returns the boundary of this element."""
            ...

        def permute(self, sigma: Any) -> P:
            """Returns the result of permuting this element by the permutation sigma."""
            ...
