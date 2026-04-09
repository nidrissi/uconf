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
        # Cache morphism results: the morphism f(p) depends only on p,
        # not on the algebra elements. We cache by the frozenset of
        # (key, coeff) pairs, which is hashable and identifies the element.
        self._morphism_cache: dict = {}

    def act(self, p_element: Any, algebra_elements: Any) -> Any:
        """Apply the pullback structure map ``γ^Q(f(p); a_1, …, a_n)``."""
        # Cache key: tuple of (basis_key, coeff) pairs, which is hashable
        cache_key = tuple(p_element)
        cached = self._morphism_cache.get(cache_key)
        if cached is not None:
            q_element = cached
        else:
            q_element = self.morphism(p_element)
            self._morphism_cache[cache_key] = q_element
        return self.algebra.act(q_element, algebra_elements)

    def boundary(self, a: Any) -> Any:
        """Apply the differential of the underlying module."""
        return self.algebra.boundary(a)
