"""Rank-1 sphere cochain models and explicit Surjection actions.

This module implements the reduced cochain complex of ``S^d`` as a
one-dimensional module and equips it with the explicit
``Surjection``-algebra structure described in ``article.tex``.
"""

from __future__ import annotations

from math import prod

from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.models.surjection import Surjection


def _permutation_sign(one_line: tuple[int, ...]) -> int:
    """Return the signature of a permutation in one-line notation."""
    inversions = 0
    for i in range(len(one_line)):
        for j in range(i + 1, len(one_line)):
            if one_line[i] > one_line[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _extract_concatenated_permutations(
    u: tuple[int, ...], n: int, d: int
) -> list[tuple[int, ...]] | None:
    """Return the permutation blocks if ``u`` has the sphere-admissible form."""
    if n <= 0:
        return None

    expected_len = n + d * (n - 1)
    if len(u) != expected_len:
        return None

    blocks = d + 1
    block_len = n - 1

    perms: list[tuple[int, ...]] = []
    pos = 0
    valid_values = set(range(1, n + 1))
    for _ in range(blocks):
        block = tuple(u[pos : pos + block_len])
        pos += block_len
        if len(set(block)) != block_len:
            return None
        if set(block) - valid_values:
            return None
        missing = tuple(valid_values - set(block))
        if len(missing) != 1:
            return None
        perms.append(block + (missing[0],))

    if pos != len(u) - 1:
        return None
    if u[pos] != perms[-1][-1]:
        return None

    for j in range(d):
        if perms[j][-1] != perms[j + 1][0]:
            return None

    return perms


def _sphere_surjection_basis_sign(u: tuple[int, ...], n: int, d: int) -> int:
    """Return the sign predicted by the proposition-level closed formula."""
    perms = _extract_concatenated_permutations(u, n=n, d=d)
    if perms is None:
        return 0

    sign_exp = d * (n * (n - 1) // 2)
    sign_exp += ((d * (d - 1)) // 2) * ((n + 2) * (n - 1) // 2)
    epsilon = -1 if sign_exp % 2 else 1
    epsilon *= prod(_permutation_sign(sigma) for sigma in perms[:d])
    return epsilon


class ReducedSphereCochains(CombinatorialFreeModule):
    """Reduced cochains of ``S^d`` as a rank-1 graded module.

    The module has one basis element ``()`` and is concentrated in degree ``d``.
    """

    def __init__(self, d: int, base_ring=QQ):
        assert d >= 0
        name = f"C~*(S^{d})"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._sphere_dim = d
        self.boundary = self.module_morphism(
            on_basis=lambda _basis: self.zero(), codomain=self
        )

    def sphere_dim(self) -> int:
        """Return the sphere dimension ``d``."""
        return self._sphere_dim

    def _element_constructor_(self, x):
        if isinstance(x, dict):
            clean = {}
            for key, coeff in x.items():
                if isinstance(key, (tuple, list)) and tuple(key) == ():
                    clean[()] = clean.get((), 0) + coeff
            return super()._element_constructor_(clean)
        if isinstance(x, (tuple, list)) and tuple(x) == ():
            return self.term(())
        raise TypeError(f"Expected () or a sparse dict, got {type(x)}.")

    def degree_on_basis(self, basis_element: tuple) -> int:
        if basis_element != ():
            raise ValueError("ReducedSphereCochains has a unique basis element ().")
        return self._sphere_dim

    def generator(self) -> "ReducedSphereCochains":
        """Return the canonical degree-``d`` generator."""
        return self(())


class SurjectionSphereCochainAlgebra(OperadAlgebra):
    """Explicit ``Surjection``-algebra structure on reduced cochains of ``S^d``."""

    def __init__(self, d: int, base_ring=QQ):
        module = ReducedSphereCochains(d=d, base_ring=base_ring)
        self._sphere_dim = d
        super().__init__(
            module=module,
            operad_cls=Surjection,
            structure_map=self._act_impl,
        )

    def _generator_coeff(self, x: ReducedSphereCochains):
        coeff = 0
        for basis, scalar in x:
            if basis == ():
                coeff += scalar
        return coeff

    def _act_impl(self, p_element: Surjection.Element, algebra_elements):
        if p_element.arity() == 0:
            return self.module.zero()

        input_scalar = prod(self._generator_coeff(a) for a in algebra_elements)
        if input_scalar == 0:
            return self.module.zero()

        coeff = 0
        n = p_element.arity()
        for basis_u, p_coeff in p_element:
            epsilon = _sphere_surjection_basis_sign(basis_u, n=n, d=self._sphere_dim)
            if epsilon:
                coeff += p_coeff * epsilon

        if coeff == 0:
            return self.module.zero()
        return input_scalar * coeff * self.module.generator()
