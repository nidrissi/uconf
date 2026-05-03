from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.morphism import OperadMorphism
from sage.all import cached_method


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

    @cached_method
    def _act_on_basis_inputs(self, p_terms: tuple, algebra_input_keys: tuple):
        """Apply the pullback action to basis inputs and cache the result."""
        arity = len(algebra_input_keys)
        base_ring = self.module.base_ring()
        p_element = self.operad_cls(arity, base_ring)._from_dict(dict(p_terms), remove_zeros=True)
        q_element = self._morphism_cache.get(p_terms)
        if q_element is None:
            q_element = self.morphism(p_element)
            self._morphism_cache[p_terms] = q_element
        act_on_basis_inputs = getattr(self.algebra, "_act_on_basis_inputs", None)
        if callable(act_on_basis_inputs):
            return act_on_basis_inputs(tuple(q_element), tuple(algebra_input_keys))
        return self.algebra.act(q_element, [self.module.term(key) for key in algebra_input_keys])

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
