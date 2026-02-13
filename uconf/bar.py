from operad import *
from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]
from sage.combinat.rooted_tree import LabelledRootedTree


class BarConstruction(CombinatorialFreeModule):
    def __init__(self, operad: OperadProtocol, n: int, base_ring=QQ):
        indices = Set(LabelledRootedTree)
        super().__init__(
            base_ring,
            indices,
            prefix=f"B({operad.name}){n}",
            category=GradedModulesWithBasis(base_ring),
        )
        self.operad = operad
        self.name = f"B({operad.name}){n}"
        self.arity = n

    def degree_on_basis(self, tree):
        """
        Degree = Sum(|p| + 1 for p in labels)
        """
        return sum(label.degree() + 1 for label in tree.labels())
