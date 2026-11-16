"""Shifted cooperad wrapper with suspension/Koszul sign conventions."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sage.all import cached_method, tensor

from uconf.core.component import ComponentBase
from uconf.core.cooperad import Cooperad, CooperadElement
from uconf.core.signs import (
    shifted_boundary_sign,
    shifted_operadic_compose_sign,
    shifted_permutation_sign,
)


class ShiftedCooperadElement(CooperadElement["ShiftedCooperadComponent"]):
    """Element wrapper carrying shifted cooperad structure maps."""

    def planarize(self):
        """Project to planar representative tensored with a group element."""
        parent = self.parent()
        return parent.planarize(self)

    def permute(self, sigma) -> ShiftedCooperadElement:
        parent = self.parent()
        sigma = parent._coerce_permutation(sigma)

        base_permuted = (
            parent.base_parent()
            .sum_of_terms((basis, parent.base_ring()(coeff)) for basis, coeff in self)
            .permute(sigma)
        )
        sign = shifted_permutation_sign(parent.factory.shift_degree, sigma)
        return parent.from_base(sign * base_permuted)

    def base_element(self):
        parent = self.parent()
        R = parent.base_ring()
        return parent.base_parent().sum_of_terms((basis, R(coeff)) for basis, coeff in self)


class ShiftedCooperadComponent(ComponentBase["ShiftedCooperadElement"]):
    """A fixed-arity component of a shifted cooperad."""

    Element = ShiftedCooperadElement

    if TYPE_CHECKING:

        @property
        def factory(self) -> ShiftedCooperad: ...

    def __init__(self, factory: ShiftedCooperad, n: int, base_ring):
        super().__init__(factory, n, base_ring)
        self._base_parent = factory.cooperad(n, base_ring)
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis,
            codomain=self,
        )

        if callable(getattr(self._base_parent, "planarize", None)):
            from sage.all import SymmetricGroupAlgebra
            from sage.all import tensor as sage_tensor

            self._pz_sga = SymmetricGroupAlgebra(base_ring, n)
            self._pz_codomain = sage_tensor([self, self._pz_sga])

            self.planarize = self.module_morphism(
                on_basis=self._planarize_on_basis,
                codomain=self._pz_codomain,
            )

    @cached_method
    def _planarize_on_basis(self, p_key):
        """Planarize a shifted-cooperad basis key.

        Applies the base cooperad's ``planarize`` and twists the resulting
        permutation sign by ``sgn(σ)^shift``.
        """
        base_pz = self._base_parent.planarize(self._base_parent(p_key))
        result = self._pz_codomain.zero()
        for (p_pl_key, sigma_key), coeff in base_pz:
            sigma = self._symmetric_group(sigma_key)
            sign_twist = int(sigma.sign()) ** self.factory.shift_degree
            result += (sign_twist * coeff) * self(p_pl_key).tensor(self._pz_sga(sigma))
        return result

    def _validate_basis_key(self, basis_key):
        return self._base_parent._validate_basis_key(basis_key)

    def _element_constructor_(self, x) -> ShiftedCooperadElement:
        if isinstance(x, ShiftedCooperadElement):
            return self.from_base(x.base_element())
        if hasattr(x, "parent") and x.parent() is self._base_parent:
            return self.from_base(x)
        return super()._element_constructor_(x)

    def _boundary_on_basis(self, basis_element):
        R = self.base_ring()
        sign = shifted_boundary_sign(self.factory.shift_degree)
        base_bdry = sign * self._base_parent.boundary(self._base_parent(basis_element))
        return self.sum_of_terms((basis, R(coeff)) for basis, coeff in base_bdry)

    def base_parent(self):
        return self._base_parent

    def from_base(self, element) -> ShiftedCooperadElement:
        R = self.base_ring()
        if element.parent() is self._base_parent:
            return self.sum_of_terms((basis, R(coeff)) for basis, coeff in element)
        base_coerced = self._base_parent.sum_of_terms((basis, R(coeff)) for basis, coeff in element)
        return self.sum_of_terms((basis, coeff) for basis, coeff in base_coerced)

    def basis_iter(self, d: int) -> Iterator[ShiftedCooperadElement]:
        """Iterate over basis elements of this shifted-cooperad component in degree ``d``.

        The arity-``n`` component of ``ShiftedCooperad(C, s)`` has its degrees
        shifted by ``s*(n-1)`` relative to the base cooperad ``C(n)``, so a
        base-cooperad element in degree ``d - s*(n-1)`` appears here in degree ``d``.
        """
        unshifted_degree = d - self.factory.shift_degree * (self._arity - 1)
        base_parent = self._base_parent
        base_basis_it = getattr(base_parent, "basis_iter", None)
        if base_basis_it is not None:
            R = self.base_ring()
            for elem in base_basis_it(unshifted_degree):
                yield self.sum_of_terms((key, R(coeff)) for key, coeff in elem)
        else:
            for key in base_parent.basis():
                if base_parent.degree_on_basis(key) == unshifted_degree:
                    yield self(key)

    def degree_on_basis(self, basis_element) -> int:
        if isinstance(basis_element, ShiftedCooperadElement):
            support = basis_element.support()
            if not support:
                return 0
            basis_element = next(iter(support))
        base_degree = self._base_parent.degree_on_basis(basis_element)
        return base_degree + self.factory.shift_degree * (self.arity() - 1)

    def _repr_term(self, basis_element) -> str:
        return self._base_parent._repr_term(basis_element)

    def _latex_term(self, basis_element) -> str:
        return self._base_parent._latex_term(basis_element)


class ShiftedCooperad(Cooperad[ShiftedCooperadComponent]):
    """The shifted cooperad ``C[d]``.

    For a cooperad ``C`` and integer shift ``d``, the arity-``n`` component is
    shifted by ``d (n - 1)``. Differential and symmetric action use the same
    transport/sign rules as :class:`uconf.shifted_operad.ShiftedOperad`.
    """

    Component = ShiftedCooperadComponent
    Element = ShiftedCooperadElement

    def __init__(self, cooperad: Cooperad, shift_degree: int):
        self.cooperad = cooperad
        self.shift_degree = int(shift_degree)
        self.name = f"{cooperad.name}[{self.shift_degree}]"
        # ``C[d](n)`` lives in degrees >= (k + d)(n - 1) where k is the
        # connectivity of C: the shift by d(n - 1) raises the lower bound.
        self.connectivity = cooperad.connectivity + self.shift_degree

    def unit_key(self) -> object:
        """Return the basis key of the counit generator in arity ``1``."""
        return self.cooperad.unit_key()

    def counit(self, x: ShiftedCooperadElement):
        parent = x.parent()
        R = parent.base_ring()
        base_x = parent.base_parent().sum_of_terms((basis, R(coeff)) for basis, coeff in x)
        return self.cooperad.counit(base_x)

    def reduced(self, x: ShiftedCooperadElement) -> ShiftedCooperadElement:
        parent = x.parent()
        R = parent.base_ring()
        base_x = parent.base_parent().sum_of_terms((basis, R(coeff)) for basis, coeff in x)
        return parent.from_base(self.cooperad.reduced(base_x))

    def infinitesimal_cocompose(self, x: ShiftedCooperadElement, i: int, m: int, n: int):
        parent = x.parent()
        left_parent = self(m, parent.base_ring())
        right_parent = self(n, parent.base_ring())
        target = tensor([left_parent, right_parent])

        R = parent.base_ring()
        base_x = parent.base_parent().sum_of_terms((basis, R(coeff)) for basis, coeff in x)
        base_delta = self.cooperad.infinitesimal_cocompose(base_x, i, m, n)

        def term_generator():
            Rt = target.base_ring()
            for key, coeff in base_delta:
                _left_basis, right_basis = key
                right_degree = right_parent.base_parent().degree_on_basis(right_basis)
                sign = shifted_operadic_compose_sign(
                    self.shift_degree,
                    i,
                    m,
                    n,
                    right_degree,
                )
                yield (key, Rt(sign * coeff))

        return target.sum_of_terms(term_generator())
