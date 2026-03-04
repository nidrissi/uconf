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

from collections.abc import Callable

from uconf.core.cooperad import CooperadProtocol


class CooperadCoalgebra:
    """A dg-module equipped with a C-coalgebra structure.

    Wraps an underlying ``CombinatorialFreeModule`` (the module V) and a
    cooperad class ``cooperad_cls`` (satisfying
    :class:`uconf.cooperad.CooperadProtocol`) together with an explicit
    costructure map.

    The costructure map can be supplied in two ways:

    1. Pass a callable as the ``costructure_map`` argument::

           coalg = CooperadCoalgebra(module, CoAssociative, my_map)

       where ``my_map(v_element, n) → (C(n) ⊗ V^{⊗n}).Element``.

    2. Subclass and override :meth:`coact`::

           class MyCoalgebra(CooperadCoalgebra):
               def coact(self, v_element, n):
                   ...

    Args:
        module: Underlying dg-module (a ``CombinatorialFreeModule``).
        cooperad_cls: Cooperad class (CooperadProtocol-compatible).
        costructure_map: Optional callable implementing the C-coalgebra
            coaction δ_n.  If omitted, a subclass must override :meth:`coact`.
    """

    def __init__(
        self,
        module,
        cooperad_cls: type[CooperadProtocol],
        costructure_map: Callable | None = None,
    ):
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

        Raises:
            NotImplementedError: If no ``costructure_map`` was given and the
                subclass has not overridden :meth:`coact`.
        """
        if self._costructure_map is not None:
            return self._costructure_map(v_element, n)
        raise NotImplementedError(
            "Provide a costructure_map to the constructor or override coact() in a subclass."
        )

    def boundary(self, v):
        """Apply the differential ∂_V to a coalgebra element.

        Args:
            v: An element of the coalgebra module.

        Returns:
            The boundary ∂_V(v) in the module.
        """
        return self.module.boundary(v)
