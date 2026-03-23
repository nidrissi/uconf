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

from collections.abc import Iterable
from typing import Any, Generic, Protocol, TypeVar

from uconf.core.cooperad import CooperadLike, CooperadComponent

CoalgebraElementType = TypeVar("CoalgebraElementType")
CooperadElementType = TypeVar("CooperadElementType", bound=CooperadComponent.Element)
CoactionValueType = TypeVar(
    "CoactionValueType",
    bound=Iterable[tuple[tuple[object, ...], object]],
    covariant=True,
)
CoalgebraElementInputType = TypeVar(
    "CoalgebraElementInputType",
    contravariant=True,
)


class CoactionMap(Protocol[CoalgebraElementInputType, CoactionValueType]):
    """Type contract for cooperad coaction callables."""

    def __call__(self, v_element: CoalgebraElementInputType, n: int, /) -> CoactionValueType: ...


class CooperadCoalgebra(Generic[CoalgebraElementType, CoactionValueType]):
    """A dg-module equipped with a C-coalgebra structure.

    Wraps an underlying ``CombinatorialFreeModule`` (the module V) and a
    cooperad provider ``cooperad_cls`` (satisfying
    :class:`uconf.cooperad.CooperadComponent`) together with an explicit
    costructure map.

    The coaction map is supplied as a callable via ``coaction_map``::

           coalg = CooperadCoalgebra(module, CoAssociative, my_map)

       where ``my_map(v_element, n) → (C(n) ⊗ V^{⊗n}).Element``.

    Args:
        module: Underlying dg-module (a ``CombinatorialFreeModule``).
        cooperad_cls: Cooperad provider (class or factory, CooperadComponent-compatible).
        coaction_map: Callable implementing the C-coalgebra coaction δ_n.

    """

    def __init__(
        self,
        module,
        cooperad_cls: CooperadLike,
        coaction_map: CoactionMap[CoalgebraElementType, CoactionValueType],
    ):
        if not callable(coaction_map):
            raise TypeError("coaction_map must be callable.")
        self.module = module
        self.cooperad_cls = cooperad_cls
        self._coaction_map = coaction_map

    def coact(self, v_element: CoalgebraElementType, n: int) -> CoactionValueType:
        """Apply the C-coaction δ_n(v) ∈ C(n) ⊗_{S_n} V^{⊗n}.

        Args:
            v_element: An element of the coalgebra module V.
            n: The coaction arity (number of factors in the coaction).

        Returns:
            An element of ``C(n) ⊗ V^{⊗n}``.

        Raises:
            ValueError: If ``n <= 0``.
            TypeError: If ``coaction_map`` does not return an iterable object.

        """
        if n <= 0:
            raise ValueError(f"Coaction arity must be positive, got {n}.")

        value = self._coaction_map(v_element, n)
        try:
            iter(value)
        except TypeError as exc:
            raise TypeError("coaction_map must return an iterable coaction value.") from exc
        return value

    def boundary(self, v):
        """Apply the differential ∂_V to a coalgebra element.

        Args:
            v: An element of the coalgebra module.

        Returns:
            The boundary ∂_V(v) in the module.

        """
        return self.module.boundary(v)

    def __call__(self, *args: Any, **kwds: Any) -> CoalgebraElementType:
        return self.module(*args, **kwds)
