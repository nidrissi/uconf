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


def _extract_concatenated_permutations(
    u: tuple[int, ...], n: int, d: int
) -> list[tuple[int, ...]] | None:
    """Return the permutation blocks if ``u`` has the sphere-admissible form.

    A surjection ``u`` of arity ``n`` and degree ``d*(n-1)`` is
    *sphere-admissible* if it can be written as the concatenation of
    ``(d+1)`` permutations ``σ_1,...,σ_{d+1}`` of ``{1,...,n}`` with
    the overlap condition ``σ_j(n) = σ_{j+1}(1)`` for ``j=1,...,d``.
    That is,
    ``u = (σ_1(1),...,σ_1(n-1), σ_2(1),...,σ_d(n-1), σ_{d+1}(1),...,σ_{d+1}(n))``.

    Returns the list of permutation blocks ``[σ_1,...,σ_{d+1}]`` if
    sphere-admissible, or ``None`` otherwise.
    """
    if n <= 0:
        return None
    if n == 1:
        # For arity 1, the only sphere-admissible element is (1,) for any d.
        if u == (1,):
            return [(1,)]
        return None

    expected_len = n + d * (n - 1)
    if len(u) != expected_len:
        return None

    perms: list[tuple[int, ...]] = []
    valid = set(range(1, n + 1))
    for i in range(0, len(u) - 1, n - 1):
        block = tuple(u[i : i + n])
        if set(block) != valid:
            return None
        perms.append(block)

    return perms


def _sphere_surjection_basis_sign(u: tuple[int, ...], n: int, d: int) -> int:
    r"""Return the Berger-Fresse sign for the ``Surjection``-algebra on ``N*(S^d)``.

    For a sphere-admissible surjection ``u`` of arity ``n`` and
    degree ``d*(n-1)``, the unique valid Alexander-Whitney cut of the
    top chain ``(0,...,d)`` has *edge factors* (1-dimensional simplex
    factors) at the ``d`` connecting positions ``j*(n-1)`` for
    ``j = 1,...,d``, and *vertex factors* (0-dimensional) elsewhere.

    The sign is ``(-1)^S`` where
    ``S = ord_inv + term1 + term2``, with:

    - ``c_j = u[j*(n-1)]`` the *connecting value* at position ``j*(n-1)``
      (equal to ``σ_j(n) = σ_{j+1}(1)`` in the permutation-block notation);
    - ``ord_inv``: the number of inversions in the sequence
      ``(c_1,...,c_d)`` (i.e. pairs ``j1 < j2`` with ``c_{j1} > c_{j2}``);
    - ``P_k = {i : u[i] = k}`` the set of positions with value ``k``;
    - ``E_{<k} = #{j : c_j < k}`` the number of connecting values
      strictly less than ``k``;
    - ``term1 = Σ_k (|P_k| - 1) · E_{<k}``;
    - ``term2 = Σ_j #{p ∈ P_{c_j} : p > j*(n-1)}``, the number of
      positions in the same group as the ``j``-th edge that come
      *after* that edge in ``u``.

    This formula is obtained by applying the Berger-Fresse action to
    the top chain of ``Δ^d`` and reading off the sign of the unique
    non-zero contribution.  The first summand ``ord_inv`` arises from
    the Koszul reordering sign, and ``term1 + term2`` arises from the
    position sign in the BF formula.

    Returns 0 for non-sphere-admissible ``u``.
    """
    if n == 1:
        # Arity-1 element (1,) acts as the identity; sign = 1.
        return 1 if u == (1,) else 0

    perms = _extract_concatenated_permutations(u, n, d)
    if perms is None:
        return 0

    if d == 0:
        # Degree-0 surjection acting on S^0: BF sign is always +1.
        return 1

    step = n - 1  # distance between consecutive connecting positions

    # Connecting values c_j = u[j*(n-1)] for j = 1,...,d
    connecting = [u[j * step] for j in range(1, d + 1)]

    # ord_inv: inversions in (c_1,...,c_d)
    ord_inv = sum(
        1
        for a in range(d)
        for b in range(a + 1, d)
        if connecting[a] > connecting[b]
    )

    # Build P[k]: positions in u whose value is k (1-indexed, use list index k)
    P: list[list[int]] = [[] for _ in range(n + 1)]
    for pos, val in enumerate(u):
        P[val].append(pos)

    # E_lt[k] = #{j : c_j < k}  (number of connecting values strictly below k)
    E_lt: list[int] = [0] * (n + 2)
    cumulative = 0
    for k in range(1, n + 1):
        E_lt[k] = cumulative
        cumulative += sum(1 for c in connecting if c == k)

    # term1: for each group k, (|P_k| - 1) * E_{<k}
    term1 = sum((len(P[k]) - 1) * E_lt[k] for k in range(1, n + 1))

    # term2: for each connecting position j*(n-1) with value c_j,
    #        count positions in group c_j that are strictly after j*(n-1)
    term2 = sum(
        sum(1 for p in P[c] if p > j * step)
        for j, c in enumerate(connecting, start=1)
    )

    total = (ord_inv + term1 + term2) % 2
    return -1 if total else 1


class ReducedSphereCochains(CombinatorialFreeModule):
    """Reduced cochains of ``S^d`` as a rank-1 graded module.

    The module has one basis element ``()`` and is concentrated in degree ``d``.
    """

    def __init__(self, d: int, base_ring=QQ):
        assert d >= 0
        name = f"N*(S^{d})"
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
