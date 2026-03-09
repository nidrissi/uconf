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

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from uconf.core.operad import OperadProtocol

AlgebraElementType = TypeVar("AlgebraElementType")
OperadElementType = TypeVar("OperadElementType", bound=OperadProtocol.Element)
OperadElementInputType = TypeVar(
    "OperadElementInputType",
    bound=OperadProtocol.Element,
    contravariant=True,
)


class StructureMap(Protocol[OperadElementInputType, AlgebraElementType]):
    """Type contract for operad action callables."""

    def __call__(
        self,
        p_element: OperadElementInputType,
        algebra_elements: Sequence[AlgebraElementType],
        /,
    ) -> AlgebraElementType: ...


class OperadAlgebra(Generic[OperadElementType, AlgebraElementType]):
    """A dg-module equipped with a P-algebra structure.

    Wraps an underlying ``CombinatorialFreeModule`` (the module A) and an
    operad class ``operad_cls`` (satisfying :class:`uconf.operad.OperadProtocol`)
    together with an explicit structure map.

    The structure map is supplied as a callable via ``structure_map``::

           alg = OperadAlgebra(module, Associative, my_map)

       where ``my_map(p_element, algebra_elements) → A.Element``.

    Args:
        module: Underlying dg-module (a ``CombinatorialFreeModule``).
        operad_cls: Operad class (OperadProtocol-compatible).
        structure_map: Callable implementing the P-algebra action γ.
    """

    def __init__(
        self,
        module,
        operad_cls: type[OperadProtocol],
        structure_map: StructureMap[OperadElementType, AlgebraElementType],
    ):
        if not callable(structure_map):
            raise TypeError("structure_map must be callable.")
        self.module = module
        self.operad_cls = operad_cls
        self._structure_map = structure_map

    def act(
        self,
        p_element: OperadElementType,
        algebra_elements: Sequence[AlgebraElementType],
    ) -> AlgebraElementType:
        """Apply the P-algebra structure map γ(p; a_1, ..., a_n).

        Args:
            p_element: An element of ``operad_cls(n)`` for some arity ``n``.
            algebra_elements: A list of ``n`` elements of the algebra module.

        Returns:
            An element of the algebra module A.

        Raises:
            TypeError: If ``p_element`` is not an element of ``operad_cls``.
            ValueError: If ``len(algebra_elements) != p_element.arity()``.
        """
        element_cls = getattr(self.operad_cls, "Element", None)
        if isinstance(element_cls, type) and not isinstance(p_element, element_cls):
            raise TypeError(
                f"Expected p_element of type {self.operad_cls.__name__}.Element, "
                f"got {type(p_element).__name__}."
            )

        n = p_element.arity()
        if len(algebra_elements) != n:
            raise ValueError(
                f"Expected {n} algebra elements for P({n}) action, "
                f"got {len(algebra_elements)}."
            )
        return self._structure_map(p_element, algebra_elements)

    def boundary(self, a: AlgebraElementType) -> AlgebraElementType:
        """Apply the differential ∂_A to an algebra element.

        Args:
            a: An element of the algebra module.

        Returns:
            The boundary ∂_A(a) in the module.
        """
        return self.module.boundary(a)
