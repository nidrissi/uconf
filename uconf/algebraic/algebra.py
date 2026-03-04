"""Algebras over dg-operads (P-algebras).

An algebra over a dg-operad P is a dg-module A together with a collection
of structure maps

    γ_n : P(n) ⊗_{S_n} A^{⊗n} → A

satisfying:
- Unit: γ_1(id; a) = a for the operadic unit id ∈ P(1).
- Composition: γ_{m+n-1}(p ∘_i q; ...) = γ_m(p; ..., γ_n(q; ...), ...).
- Equivariance: γ_n(p·σ; a_1,...,a_n) = γ_n(p; a_{σ^{-1}(1)}, ..., a_{σ^{-1}(n)}).
- Chain map: ∂_A ∘ γ_n = γ_n ∘ (∂_P ⊗ id + ...) (Leibniz rule).

Reference: Loday-Vallette "Algebraic Operads", Chapter 12.
"""

from __future__ import annotations

from typing import Callable

from uconf.core.operad import OperadProtocol


class OperadAlgebra:
    """A dg-module equipped with a P-algebra structure.

    Wraps an underlying ``CombinatorialFreeModule`` (the module A) and an
    operad class ``operad_cls`` (satisfying :class:`uconf.operad.OperadProtocol`)
    together with an explicit structure map.

    The structure map is provided as a callable::

        structure_map(p_element, algebra_elements) → A.Element

    where ``p_element`` is an element of ``operad_cls(n)`` for some ``n``, and
    ``algebra_elements`` is a list of ``n`` elements of the module A.

    Args:
        module: Underlying dg-module (a ``CombinatorialFreeModule``).
        operad_cls: Operad class (OperadProtocol-compatible).
        structure_map: Callable implementing the P-algebra action γ.
    """

    def __init__(
        self, module, operad_cls: type[OperadProtocol], structure_map: Callable
    ):
        self.module = module
        self.operad_cls = operad_cls
        self._structure_map = structure_map

    def act(self, p_element: OperadProtocol.Element, algebra_elements: list):
        """Apply the P-algebra structure map γ(p; a_1, ..., a_n).

        Args:
            p_element: An element of ``operad_cls(n)`` for some arity ``n``.
            algebra_elements: A list of ``n`` elements of the algebra module.

        Returns:
            An element of the algebra module A.

        Raises:
            ValueError: If ``len(algebra_elements) != p_element.arity()``.
        """
        n = p_element.arity()
        alg_list = list(algebra_elements)
        if len(alg_list) != n:
            raise ValueError(
                f"Expected {n} algebra elements for P({n}) action, got {len(alg_list)}."
            )
        return self._structure_map(p_element, alg_list)

    def boundary(self, a):
        """Apply the differential ∂_A to an algebra element.

        Args:
            a: An element of the algebra module.

        Returns:
            The boundary ∂_A(a) in the module.
        """
        return self.module.boundary(a)
