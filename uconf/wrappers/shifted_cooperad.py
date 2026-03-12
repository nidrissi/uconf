"""Shifted cooperad wrapper with suspension/Koszul sign conventions."""

from __future__ import annotations


from sage.all import (
    CombinatorialFreeModule,
    GradedModulesWithBasis,
    QQ,
    SymmetricGroup,
    UniqueRepresentation,
    tensor,
)

from uconf.core.cooperad import CooperadLike
from uconf.core.signs import (
    shifted_boundary_sign,
    shifted_operadic_compose_sign,
    shifted_permutation_sign,
)


class ShiftedCooperad(UniqueRepresentation):
    """Factory for shifted cooperad components.

    For a cooperad ``C`` and integer shift ``d``, the arity-``n`` component is
    shifted by ``d (n - 1)``. Differential and symmetric action use the same
    transport/sign rules as :class:`uconf.shifted_operad.ShiftedOperad`.
    """

    def __init__(self, cooperad_cls: CooperadLike, shift_degree: int):
        self.cooperad_cls = cooperad_cls
        self.shift_degree = int(shift_degree)
        self.name = f"{cooperad_cls.name}[{self.shift_degree}]"

    def __call__(self, n: int, base_ring=QQ) -> "ShiftedCooperad.Component":
        return ShiftedCooperad.Component(self, n, base_ring)

    class Component(CombinatorialFreeModule):
        """A fixed-arity component of a shifted cooperad."""

        def __init__(self, factory: "ShiftedCooperad", n: int, base_ring=QQ):
            assert n >= 0, f"Arity must be non-negative. Got {n}."
            name = f"{factory.name}{n}"
            super().__init__(
                base_ring,
                tuple,
                prefix=name,
                category=GradedModulesWithBasis(base_ring),
            )
            self.factory = factory
            self._arity = int(n)
            self._base_parent = factory.cooperad_cls(n, base_ring)
            self._symmetric_group = SymmetricGroup(self._arity)
            self.rename(name)
            self.boundary = self.module_morphism(
                on_basis=self._boundary_on_basis,
                codomain=self,
            )

        def _validate_basis_key(self, basis_key):
            if hasattr(self._base_parent, "_validate_basis_key"):
                return self._base_parent._validate_basis_key(basis_key)
            return basis_key

        def _element_constructor_(self, x):
            if isinstance(x, ShiftedCooperad.Element):
                return self.from_base(x.base_element())

            if hasattr(x, "parent") and x.parent() is self._base_parent:
                return self.from_base(x)

            if isinstance(x, dict):
                clean_dict = {}
                for key, coeff in x.items():
                    clean_key = self._validate_basis_key(key)
                    if clean_key is None:
                        continue
                    clean_dict[clean_key] = coeff
                return super()._element_constructor_(clean_dict)

            if isinstance(x, (tuple, list)):
                clean_key = self._validate_basis_key(x)
                if clean_key is None:
                    return self.zero()
                return self.term(clean_key)

            return super()._element_constructor_(x)

        def _boundary_on_basis(self, basis_element):
            sign = shifted_boundary_sign(self.factory.shift_degree)
            base_bdry = sign * self._base_parent.boundary(
                self._base_parent.term(basis_element)
            )
            return self.sum_of_terms((basis, coeff) for basis, coeff in base_bdry)

        def arity(self) -> int:
            return self._arity

        def base_ring(self):
            return self._base_parent.base_ring()

        def base_parent(self):
            return self._base_parent

        def from_base(self, element) -> "ShiftedCooperad.Element":
            if element.parent() is self._base_parent:
                return self.sum_of_terms((basis, coeff) for basis, coeff in element)
            base_coerced = self._base_parent.sum_of_terms(
                (basis, coeff) for basis, coeff in element
            )
            return self.sum_of_terms((basis, coeff) for basis, coeff in base_coerced)

        def degree_on_basis(self, basis_element) -> int:
            if isinstance(basis_element, ShiftedCooperad.Element):
                support = basis_element.support()
                if not support:
                    return 0
                basis_element = next(iter(support))
            base_degree = self._base_parent.degree_on_basis(basis_element)
            return base_degree + self.factory.shift_degree * (self.arity() - 1)

        def counit(self, x: "ShiftedCooperad.Element"):
            base_x = self.base_parent().sum_of_terms(
                (basis, coeff) for basis, coeff in x
            )
            return self.factory.cooperad_cls.counit(base_x)

        def reduced(self, x: "ShiftedCooperad.Element") -> "ShiftedCooperad.Element":
            base_x = self.base_parent().sum_of_terms(
                (basis, coeff) for basis, coeff in x
            )
            return self.from_base(self.factory.cooperad_cls.reduced(base_x))

        def infinitesimal_cocompose(
            self, x: "ShiftedCooperad.Element", i: int, m: int, n: int
        ):
            left_parent = self.factory(m, self.base_ring())
            right_parent = self.factory(n, self.base_ring())
            target = tensor([left_parent, right_parent])

            base_x = self.base_parent().sum_of_terms(
                (basis, coeff) for basis, coeff in x
            )
            base_delta = self.factory.cooperad_cls.infinitesimal_cocompose(
                base_x, i, m, n
            )

            def term_generator():
                for key, coeff in base_delta:
                    left_basis, right_basis = key
                    right_degree = right_parent.base_parent().degree_on_basis(
                        right_basis
                    )
                    sign = shifted_operadic_compose_sign(
                        self.factory.shift_degree,
                        i,
                        m,
                        n,
                        right_degree,
                    )
                    yield (key, sign * coeff)

            return target.sum_of_terms(term_generator())

    class Element(CombinatorialFreeModule.Element):
        """Element wrapper carrying shifted cooperad structure maps."""

        def arity(self) -> int:
            return self.parent().arity()

        def boundary(self) -> "ShiftedCooperad.Element":
            return self.parent().boundary(self)

        def permute(self, sigma) -> "ShiftedCooperad.Element":
            if isinstance(sigma, (list, tuple)):
                sigma = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity()}. Got {sigma} ({type(sigma)})."
                )

            base_permuted = (
                self.parent()
                .base_parent()
                .sum_of_terms((basis, coeff) for basis, coeff in self)
                .permute(sigma)
            )
            sign = shifted_permutation_sign(self.parent().factory.shift_degree, sigma)
            return self.parent().from_base(sign * base_permuted)

        def counit(self):
            return self.parent().counit(self)

        def reduced(self) -> "ShiftedCooperad.Element":
            return self.parent().reduced(self)

        def infinitesimal_cocompose(self, i: int, m: int, n: int):
            return self.parent().infinitesimal_cocompose(self, i, m, n)

        def base_element(self):
            return (
                self.parent()
                .base_parent()
                .sum_of_terms((basis, coeff) for basis, coeff in self)
            )


ShiftedCooperad.Component.Element = ShiftedCooperad.Element
