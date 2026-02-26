from typing import ClassVar

from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class Lie(CombinatorialFreeModule):
    name: ClassVar[str] = "Lie"

    def __init__(self, n, base_ring=QQ):
        assert n >= 0, f"Arity must be non-negative. Got {n}."
        name = f"{self.name}{n}"
        raise NotImplementedError()

    def _element_constructor_(self, x):
        raise NotImplementedError()

    def arity(self) -> int:
        return self._arity

    @staticmethod
    def unit():
        raise NotImplementedError()

    def degree_on_basis(self, element) -> int:
        raise NotImplementedError()

    @staticmethod
    def compose(x: Lie, i: int, y: Lie) -> Lie:
        raise NotImplementedError()

    class Element(CombinatorialFreeModule.Element):
        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> Lie.Element:
            return self.parent().boundary(self)

        def permute(self, sigma: Permutation) -> Lie.Element:
            raise NotImplementedError()
