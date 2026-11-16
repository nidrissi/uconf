"""Shifted operad construction with standard Koszul/sign-representation twists.

For an operad ``P`` and integer shift ``d``, this module implements a wrapper
``ShiftedOperad(P, d)`` whose arity-``n`` component is shifted by ``d (n - 1)``.
The symmetric-group action and composition follow the usual suspension-style
sign rules from standard operad references.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from sage.all import cached_method

from uconf.core.component import ComponentBase, ElementBase
from uconf.core.operad import Operad
from uconf.core.signs import (
    shifted_boundary_sign,
    shifted_operadic_compose_sign,
    shifted_permutation_sign,
)


class ShiftedOperadElement(ElementBase["ShiftedOperadComponent"]):
    """Element wrapper carrying shifted signs and structure maps.

    The stored basis keys are the same as in the underlying operad; the
    shift only changes degrees and the signs in boundary, permutation, and
    composition formulas.
    """

    def planarize(self):
        """Project to planar representative tensored with a group element."""
        parent = self.parent()
        return parent.planarize(self)

    def permute(self, sigma) -> ShiftedOperadElement:
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
        """Return the underlying element in the base operad component."""
        parent = self.parent()
        R = parent.base_ring()
        return parent.base_parent().sum_of_terms((basis, R(coeff)) for basis, coeff in self)

    def underlying_element(self):
        """Return the underlying element in the base operad component."""
        return self.base_element()


class ShiftedOperadComponent(ComponentBase["ShiftedOperadElement"]):
    """A fixed-arity component of a shifted operad."""

    Element = ShiftedOperadElement

    if TYPE_CHECKING:

        @property
        def factory(self) -> ShiftedOperad: ...

    def __init__(self, factory: ShiftedOperad, n: int, base_ring):
        super().__init__(factory, n, base_ring)
        self._base_parent = factory.operad_cls(n, base_ring)
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis,
            codomain=self,
        )

        # Expose planar_basis_iter when the base operad is quasi-planar.
        # This makes ShiftedOperad wrapping a quasi-planar operad also
        # quasi-planar, so that FreeAlgebraModule.basis_iter() can correctly
        # enumerate S_n-orbit representatives via the isomorphism
        # P(n) ⊗_{S_n} M^{⊗n} ≅ P_pl(n) ⊗ M^{⊗n}.
        if hasattr(self._base_parent, "planar_basis_iter"):
            shift = factory.shift_degree
            arity = int(n)
            base_parent = self._base_parent
            self_ref = self

            def _planar_basis_it(d):
                unshifted = d - shift * (arity - 1)
                R = self_ref.base_ring()
                for elem in base_parent.planar_basis_iter(unshifted):
                    yield self_ref.sum_of_terms((k, R(c)) for k, c in elem)

            self.planar_basis_iter = _planar_basis_it

        # Expose planarize when the base operad has planarize.
        # The S_n-action on ShiftedOperad(P, d)(n) is twisted by sgn(σ)^d,
        # so the normalisation identity becomes:
        #   (p ·_base σ, m_tuple) = sgn(σ)^d · (p_planar, σ · m_tuple)
        # in the coinvariant quotient ShiftedP(n) ⊗_{S_n} M^n.
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
        """Planarize a single shifted-operad basis key.

        Applies the base operad's ``planarize`` and twists the resulting
        permutation sign by ``sgn(σ)^shift``.  Cached per basis key so
        that repeated calls during basis enumeration are O(1).
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

    def _element_constructor_(self, x) -> ShiftedOperadElement:
        if isinstance(x, ShiftedOperadElement):
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

    def underlying_parent(self):
        """Return the underlying base-operad Sage parent in this arity."""
        return self._base_parent

    def basis_iter(self, d: int) -> Iterator[ShiftedOperadElement]:
        """Iterate over basis elements of this shifted-operad component in degree ``d``.

        The arity-``n`` component of ``ShiftedOperad(P, s)`` has its degrees
        shifted by ``s*(n-1)`` relative to the base operad ``P(n)``, so a
        base-operad element in degree ``d - s*(n-1)`` appears here in degree ``d``.
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

    def from_base(self, element) -> ShiftedOperadElement:
        R = self.base_ring()
        if element.parent() is self._base_parent:
            return self.sum_of_terms((basis, R(coeff)) for basis, coeff in element)
        else:
            base_coerced = self._base_parent.sum_of_terms(
                (basis, R(coeff)) for basis, coeff in element
            )
            return self.sum_of_terms((basis, coeff) for basis, coeff in base_coerced)

    def degree_on_basis(self, basis_element) -> int:
        if isinstance(basis_element, ShiftedOperadElement):
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


class ShiftedOperad(Operad[ShiftedOperadComponent]):
    r"""The shifted operad ``P[d]``.

    Args:
        operad_cls: Base operad class (e.g. ``Lie`` or ``Surjection``).
        shift_degree: Integer ``d`` in the arity-dependent shift ``d (n-1)``.

    The construction uses the following conventions for homogeneous terms:

    - Symmetric action twist by ``sgn(sigma)^d``.
    - Composition twist
      ``(-1)^(d * ((i-1)(n-1) + (m-1)|y|))``
      with ``m = arity(x)``, ``n = arity(y)``, and ``|y|`` in the base operad.

    EXAMPLES::

        sage: from sage.all import QQ
        sage: from uconf import Lie, ShiftedOperad
        sage: sLie = ShiftedOperad(Lie, 1)
        sage: x = sLie(2, QQ)((1,))
        sage: x.arity()
        2

    """

    Component = ShiftedOperadComponent
    Element = ShiftedOperadElement

    def __init__(self, operad_cls: Operad, shift_degree: int):
        self.operad_cls = operad_cls
        self.shift_degree = int(shift_degree)
        self.name = f"{operad_cls.name}[{self.shift_degree}]"
        # ``P[d](n)`` lives in degrees >= (k + d)(n - 1) where k is the
        # connectivity of P: the shift by d(n - 1) raises the lower bound.
        self.connectivity = operad_cls.connectivity + self.shift_degree

    def unit(self, base_ring) -> ShiftedOperadElement:
        """Return the shifted operadic unit over ``base_ring``."""
        component = self(1, base_ring)
        return component.from_base(self.operad_cls.unit(base_ring))

    def unit_key(self) -> object:
        """Return the basis key of the unit element in arity ``1``."""
        return self.operad_cls.unit_key()

    def compose(
        self, x: ShiftedOperadElement, i: int, y: ShiftedOperadElement
    ) -> ShiftedOperadElement:
        """Compose two shifted-operad elements in this wrapper."""
        x_parent = x.parent()
        y_parent = y.parent()

        if x_parent.factory is not self or y_parent.factory is not self:
            raise TypeError("Both elements must belong to this shifted operad.")
        if x_parent.base_ring() != y_parent.base_ring():
            raise TypeError("Both elements must have the same base ring.")

        m = x_parent.arity()
        n = y_parent.arity()
        target = self(m + n - 1, x_parent.base_ring())
        accumulated = target.zero()
        R = target.base_ring()

        for x_basis, x_coeff in x:
            x_term = x_parent.base_parent()(x_basis)
            for y_basis, y_coeff in y:
                y_term = y_parent.base_parent()(y_basis)
                y_degree = y_parent.base_parent().degree_on_basis(y_basis)
                base_composed = self.operad_cls.compose(x_term, i, y_term)
                sign = shifted_operadic_compose_sign(
                    self.shift_degree,
                    i,
                    m,
                    n,
                    y_degree,
                )
                accumulated += target.sum_of_terms(
                    (
                        basis,
                        R(sign * x_coeff * y_coeff * coeff),
                    )
                    for basis, coeff in base_composed
                )

        return target.from_base(accumulated)
