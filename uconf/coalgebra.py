"""Coalgebras over dg-cooperads (C-coalgebras).

A coalgebra over a dg-cooperad C is a dg-module V together with a collection
of costructure maps

    δ_n : V → C(n) ⊗_{S_n} V^{⊗n}

satisfying:
- Counit: (ε ⊗ id) ∘ δ_1 = id where ε : C(1) → k is the cooperadic counit.
- Cocomposition: coassociativity dual to the operad composition axioms.
- Coequivariance: δ_n is S_n-equivariant.
- Chain map: ∂_V ∘ δ_n = δ_n ∘ ∂_V (Leibniz rule with cooperad boundary).

Reference: Loday-Vallette "Algebraic Operads", Chapter 12.
"""

from __future__ import annotations

from typing import Callable


class CooperadCoalgebra:
    """A dg-module equipped with a C-coalgebra structure.

    Wraps an underlying ``CombinatorialFreeModule`` (the module V) and a
    cooperad class ``cooperad_cls`` (satisfying
    :class:`uconf.cooperad.CooperadProtocol`) together with an explicit
    costructure map.

    The costructure map is provided as a callable::

        costructure_map(v_element, n) → (C(n) ⊗ V^{⊗n}).Element

    where ``v_element`` is an element of the module V and ``n`` is the
    coaction arity.

    Args:
        module: Underlying dg-module (a ``CombinatorialFreeModule``).
        cooperad_cls: Cooperad class (CooperadProtocol-compatible).
        costructure_map: Callable implementing the C-coaction δ.
    """

    def __init__(self, module, cooperad_cls, costructure_map: Callable):
        self.module = module
        self.cooperad_cls = cooperad_cls
        self._costructure_map = costructure_map

    def coact(self, v_element, n: int):
        """Apply the C-coaction δ_n(v) ∈ C(n) ⊗_{S_n} V^{⊗n}.

        Args:
            v_element: An element of the coalgebra module V.
            n: The coaction arity (number of factors in the coaction).

        Returns:
            An element of ``C(n) ⊗ V^{⊗n}``.
        """
        return self._costructure_map(v_element, n)

    def boundary(self, v):
        """Apply the differential ∂_V to a coalgebra element.

        Args:
            v: An element of the coalgebra module.

        Returns:
            The boundary ∂_V(v) in the module.
        """
        return self.module.boundary(v)
