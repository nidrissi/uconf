from operad import *
from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class BarConstruction(CombinatorialFreeModule):
    def __init__(self, operad: type[OperadProtocol], n: int, base_ring=QQ):
        super().__init__(
            base_ring,
            LabelledOrderedTrees(),
            prefix=f"B({operad.name}){n}",
            category=GradedModulesWithBasis(base_ring),
        )
        self._operad = operad
        self.name = f"B({operad.name}){n}"
        self._arity = n
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

    def _element_constructor_(self, x):
        return super()._element_constructor_(x)

    def arity(self) -> int:
        return self._arity

    def degree_on_basis(self, tree: LabelledOrderedTree) -> int:
        """
        Degree = Sum(|p| + 1 for p in labels)
        """

        def term_generator():
            for label in tree.labels():
                if label is None:
                    raise ValueError("Labels of the tree cannot be None.")
                yield label.degree() + 1

        return sum(term_generator())

    def _boundary_on_basis(
        self, tree: LabelledOrderedTree
    ) -> "BarConstruction.Element":
        raise NotImplementedError("Boundary computation is not implemented yet.")

    class Element(CombinatorialFreeModule.Element):
        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "BarConstruction.Element":
            return self.parent().boundary(self)
