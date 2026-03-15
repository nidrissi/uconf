"""Morphisms of dg-operads and pullback algebras.

An operad morphism ``f: P → Q`` is a collection of linear maps
``f_n: P(n) → Q(n)`` satisfying:

- **Unit**: ``f_1(id_P) = id_Q``
- **Composition**: ``f(x ∘_i y) = f(x) ∘_i f(y)``
- **Equivariance**: ``f(x·σ) = f(x)·σ``
- **Chain map**: ``f(∂x) = ∂f(x)``

Given a ``Q``-algebra ``(A, γ^Q)`` and a morphism ``f: P → Q``, the
**pullback** ``f^*(A)`` is the ``P``-algebra ``(A, γ^P)`` with structure map
``γ^P(p; a_1,…,a_n) = γ^Q(f(p); a_1,…,a_n)``.
"""

from __future__ import annotations

from typing import Any, Callable

from uconf.core.operad import OperadLike


class OperadMorphism:
    """A morphism ``f: P → Q`` of dg-operads.

    Parameters
    ----------
    source : OperadLike
        Source operad (class or factory).
    target : OperadLike
        Target operad (class or factory).
    on_element : callable
        A function mapping elements of ``P(n)`` to elements of ``Q(n)``.
        Must be linear in the element.
    """

    def __init__(
        self,
        source: OperadLike,
        target: OperadLike,
        on_element: Callable[[Any], Any],
    ):
        self.source = source
        self.target = target
        self._on_element = on_element

    def __call__(self, element: Any) -> Any:
        """Apply the morphism to an element of the source operad."""
        return self._on_element(element)


class PullbackAlgebra:
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
