r"""Rank-1 sphere cochain models and explicit Surjection actions.

This module implements the reduced cochain complex of ``S^d`` as a
one-dimensional module and equips it with the explicit
``Surjection``-algebra structure.

**Background.**
By the Berger–Fresse theorem [BF04], the normalized cochain complex
`C^*(\Delta^N; k)` is a module over the Surjection operad.  For the
sphere `S^d`, the reduced cochain complex `N^*(S^d)` is the one-dimensional
module with generator in degree `d`.

A surjection `u \in S(n)` can act non-trivially on `N^*(S^d)` only if
its degree equals `d(n-1)` (so that the total input degree `dn` matches
the output degree `d`).  Such surjections are *sphere-admissible*: they
decompose as the concatenation of `(d+1)` permutations of `\{1,\ldots,n\}`
with consecutive overlap of one entry (see
:func:`_extract_concatenated_permutations`).

The sign of the action `\mu_u(g, \ldots, g) = \varepsilon(u) \cdot g` is
read off from the unique non-zero AW contribution when applying the BF
chain action to the top simplex `(0, \ldots, d)` (see
:func:`_sphere_surjection_basis_sign`).

**Reference**: C. Berger, B. Fresse, "Combinatorial operad actions on
cochains", Math. Proc. Cambridge Philos. Soc. **137** (2004), 135–174.
"""

from __future__ import annotations

from math import prod

from sage.all import QQ, CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.algebra import OperadAlgebra
from uconf.models.surjection import Surjection


def _extract_concatenated_permutations(
    u: tuple[int, ...], n: int, d: int
) -> list[tuple[int, ...]] | None:
    r"""Return the permutation blocks if ``u`` has the sphere-admissible form.

    A surjection ``u`` of arity ``n`` and degree ``d*(n-1)`` is
    *sphere-admissible* if it can be written as the concatenation of
    ``(d+1)`` permutations ``σ_1,...,σ_{d+1}`` of ``{1,...,n}`` with
    the overlap condition ``σ_j(n) = σ_{j+1}(1)`` for ``j=1,...,d``.
    That is,

    .. math::
        u = (\sigma_1(1),\ldots,\sigma_1(n-1),\;
             \sigma_2(1),\ldots,\sigma_2(n-1),\;
             \ldots,\;
             \sigma_{d+1}(1),\ldots,\sigma_{d+1}(n))

    so ``u`` has length ``n + d*(n-1)``, and consecutive blocks of length
    ``n`` overlap in one position.

    Returns the list of permutation blocks ``[σ_1,...,σ_{d+1}]`` if
    sphere-admissible, or ``None`` otherwise.

    **Example** (``n=2``, ``d=3``)::

        # For n=2, d=3: expected length = 2 + 3*(2-1) = 5.
        _extract_concatenated_permutations((1, 2, 1, 2, 1), n=2, d=3)
        # → [(1, 2), (2, 1), (1, 2), (2, 1)]
        # Block 0: u[0:2]=(1,2) ✓, block 1: u[1:3]=(2,1) ✓,
        # block 2: u[2:4]=(1,2) ✓, block 3: u[3:5]=(2,1) ✓.
        # Overlap: u[1]=2 (last of block 0) = u[1]=2 (first of block 1) ✓,
        #          u[2]=1 = u[2]=1 ✓, u[3]=2 = u[3]=2 ✓.

        # For n=2, d=2: expected length = 2 + 2*(2-1) = 4.
        _extract_concatenated_permutations((1, 2, 1, 1), n=2, d=2)
        # → None  (u[2:4]=(1,1) is not a permutation of {1,2})

        _extract_concatenated_permutations((1, 2, 1, 2), n=2, d=2)
        # → [(1, 2), (2, 1), (1, 2)]  (d+1=3 blocks ✓)
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
      ``(c_1,...,c_d)`` (i.e. pairs ``j1 < j2`` with ``c_{j1} > c_{j2}``),
      coming from the **ordering sign** of :func:`~uconf.algebraic.simplicial._compute_bf_sign`
      (only the edge factors have non-zero degree, so only inversions
      involving two edge positions contribute to the Koszul sign);
    - ``P_k = {i : u[i] = k}`` the set of positions with value ``k``;
    - ``E_{<k} = #{j : c_j < k}`` the number of connecting values
      strictly less than ``k``;
    - ``term1 = Σ_k (|P_k| - 1) · E_{<k}``, coming from the **action sign**
      contribution of sorting equal-valued positions relative to the
      edge factors;
    - ``term2 = Σ_j #{p ∈ P_{c_j} : p > j*(n-1)}``, the number of
      positions in the same group as the ``j``-th edge that come
      *after* that edge in ``u``, also part of the **action sign**.

    This formula is obtained by applying the Berger-Fresse action to
    the top chain of ``Δ^d`` and reading off the sign of the unique
    non-zero contribution.  The first summand ``ord_inv`` arises from
    the Koszul reordering sign, and ``term1 + term2`` arises from the
    position sign in the BF formula (see
    :func:`~uconf.algebraic.simplicial._compute_bf_sign`).

    Returns 0 for non-sphere-admissible ``u``.

    **Worked example** (``n=2``, ``d=1``):

    Consider ``u = (2, 1, 2)`` with ``n=2`` and ``d=1`` (degree
    ``1*(2-1) = 1``).

    - Step = ``n-1 = 1``.  Connecting position: ``j=1``, position ``1``.
      ``c_1 = u[1] = 1``.
    - ``ord_inv``: only one connecting value, no pairs → ``ord_inv = 0``.
    - ``P[1] = [1]``, ``P[2] = [0, 2]``.
    - ``E_lt[1] = 0``, ``E_lt[2] = #{c_j < 2} = #{1} = 1``.
    - ``term1 = (|P[1]|-1)*0 + (|P[2]|-1)*1 = 0 + 1 = 1``.
    - ``term2``: ``j=1``, ``c_1=1``, ``P[1]=[1]``, ``#{p in [1]: p > 1} = 0``.
      ``term2 = 0``.
    - ``S = 0 + 1 + 0 = 1`` → sign ``= -1``.

    Verification: ``θ_{(2,1,2)}((0,1)) = -1 · (0,1) ⊗ (0,1)`` (by a
    direct BF computation), so ``μ_{(2,1,2)}(g, g) = -g`` on ``N*(S^1)``.

    Compare ``u = (1, 2, 1)`` (same ``n=2``, ``d=1``): ``c_1 = u[1] = 2``,
    ``term1 = (2-1)*0 = 0``, ``term2 = #{p in P[2]=[1]: p > 1} = 0``,
    ``S = 0`` → sign ``= +1``.
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

    # The unique valid AW cut of the top chain (0,...,d) into n+d*(n-1) factors
    # has exactly d factors of dimension 1 (the "edge factors") located at the
    # connecting positions j*(n-1), and all other factors of dimension 0
    # (the "vertex factors").  Only edge factors have non-zero degree, so the
    # ordering sign (Koszul sign of sorting by u-value) simplifies to counting
    # inversions only among the d connecting values.

    # Connecting values c_j = u[j*(n-1)] for j = 1,...,d.
    # These are the u-values of the edge factors; vertex factors have degree 0
    # and do not contribute to the Koszul sign.
    connecting = [u[j * step] for j in range(1, d + 1)]

    # ord_inv: number of inversions in (c_1,...,c_d).
    # This is the ordering sign contribution from _compute_bf_sign, restricted
    # to pairs of edge-factor positions (all vertex-factor cross-terms vanish
    # because vertex factors have degree 0).
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

    # term1 and term2 together make up the action sign from _compute_bf_sign.
    # After sorting positions by u-value, consecutive positions of the same value
    # will be concatenated.  The action sign accumulates the cumulative degree of
    # all preceding sorted factors at each same-value adjacency.  Since only edge
    # positions contribute non-zero degree (1), this decomposes into two parts:

    # term1: for each group k with |P_k| occurrences, the E_{<k} edge positions
    # (which have been sorted before the entire k-group) each contribute 1 to
    # the action sign.  Concretely: within the sorted list, the E_{<k} edge
    # factors of values < k precede the k-group, and each consecutive pair
    # inside the k-group adds one unit of cumulative weight per edge before it.
    term1 = sum((len(P[k]) - 1) * E_lt[k] for k in range(1, n + 1))

    # term2: for each connecting position j*(n-1) with value c_j, count the
    # positions in group c_j that are sorted *after* the j-th edge position in
    # the sorted order.  These pairs (edge j, later same-value position) each
    # contribute one unit of degree-1 to the cumulative action sign.
    term2 = sum(
        sum(1 for p in P[c] if p > j * step)
        for j, c in enumerate(connecting, start=1)
    )

    total = (ord_inv + term1 + term2) % 2
    return -1 if total else 1


class ReducedSphereCochains(CombinatorialFreeModule):
    r"""Reduced cochains of ``S^d`` as a rank-1 graded module.

    The reduced cochain complex `N^*(S^d; k)` is the one-dimensional
    `k`-module spanned by the fundamental class `g` in degree `d` (using
    the homological grading convention `\deg(g) = d`).  The boundary is
    zero.

    The unique basis element is represented by the empty tuple ``()``.
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
    r"""Explicit ``Surjection``-algebra structure on reduced cochains of ``S^d``.

    For a surjection `u \in S(n)` of degree `d(n-1)` and the generator
    `g \in N^*(S^d)`, the algebra structure is

    .. math::
        \mu_u(g, \ldots, g) = \varepsilon(u) \cdot g

    where `\varepsilon(u) \in \{-1, 0, +1\}` is computed by
    :func:`_sphere_surjection_basis_sign`.  For surjections of any other
    degree, the action is zero (degree reasons).

    This is the explicit sphere model of the Berger–Fresse algebra
    structure [BF04], obtained by applying the general cochain action of
    :func:`~uconf.algebraic.simplicial.surjection_cochain_action` to the
    top cochain of `\Delta^d` and restricting to the sphere quotient.
    """

    def __init__(self, d: int, base_ring=QQ):
        module = ReducedSphereCochains(d=d, base_ring=base_ring)
        self._sphere_dim = d
        super().__init__(
            module=module,
            operad_cls=Surjection,
            structure_map=self._act_impl,
        )

    def _generator_coeff(self, x: ReducedSphereCochains):
        """Return the scalar coefficient of the generator ``()`` in ``x``."""
        coeff = 0
        for basis, scalar in x:
            if basis == ():
                coeff += scalar
        return coeff

    def _act_impl(self, p_element: Surjection.Element, algebra_elements):
        r"""Implement `\mu_u(g, \ldots, g) = \varepsilon(u) \cdot g`.

        Since `N^*(S^d)` is rank-1, every input is a scalar multiple of
        the generator `g`.  The overall scalar factor splits as:

        1. ``input_scalar``: the product of the scalar coefficients of
           all inputs (from ``algebra_elements``).
        2. For each basis surjection ``basis_u`` of ``p_element``:
           ``epsilon = _sphere_surjection_basis_sign(basis_u, n, d)``
           (zero for non-sphere-admissible surjections, otherwise ±1).

        The result is ``input_scalar * (Σ_u p_coeff_u * epsilon_u) * g``.
        """
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
