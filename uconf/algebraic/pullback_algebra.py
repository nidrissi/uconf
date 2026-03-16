from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.morphism import OperadMorphism


from typing import Any


class PullbackAlgebra(OperadAlgebra):
    """Pull back an algebra along an operad morphism.

    Given a ``Q``-algebra ``algebra`` and a morphism ``f: P → Q``,
    the pullback is the ``P``-algebra with the same underlying module
    and structure map ``γ^P(p; a_1,…,a_n) = γ^Q(f(p); a_1,…,a_n)``.

    Parameters
    ----------
    morphism : OperadMorphism
        A morphism ``f: P → Q``.
    algebra : OperadAlgebra
        A ``Q``-algebra.
    """

    def __init__(self, morphism: OperadMorphism, algebra: Any):
        self.morphism = morphism
        self.algebra = algebra
        self.module = algebra.module
        self.operad_cls = morphism.source

    def act(self, p_element: Any, algebra_elements: Any) -> Any:
        """Apply the pullback structure map ``γ^Q(f(p); a_1, …, a_n)``."""
        q_element = self.morphism(p_element)
        return self.algebra.act(q_element, algebra_elements)

    def boundary(self, a: Any) -> Any:
        """Apply the differential of the underlying module."""
        return self.algebra.boundary(a)
