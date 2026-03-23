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

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, cached_method, Family

from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.display import latex_linear_combination
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


def _permutation_sign(perm: tuple[int, ...]) -> int:
    """Return the sign of a permutation."""
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _sphere_surjection_basis_sign(u: tuple[int, ...], n: int, d: int) -> int:
    r"""
    Return the Berger-Fresse sign for the ``Surjection``-algebra on ``N*(S^d)``.

    Computes the sign associated with a basis element of the surjection algebra
    acting on the normalized chains of a sphere, following the formula from
    Proposition prop:surj-alg-sphere in article.tex.

    Parameters
    ----------
    u : tuple[int, ...]
        The arity tuple representing the surjection basis element.
    n : int
        The arity (number of inputs).
    d : int
        The dimension of the sphere.

    Returns
    -------
    int
        The Berger-Fresse sign: +1, -1, or 0 if the configuration is invalid.

        - Returns 1 if n == 1 and u == (1,) (identity element).
        - Returns 0 if n == 1 and u != (1,), or if permutations cannot be extracted.
        - Returns 1 if d == 0 (degree-0 surjection on S^0).
        - Otherwise, returns the product of sign contributions from the surjection
          structure and the extracted permutations.

    Notes
    -----
    The sign is computed as:
        - Initial sign based on degrees: (-1)^(d*n*(n-1)/2 + (d*(d-1)/2)*((n-2)*(n-1)/2))
        - Multiplied by the signs of d extracted permutations.

    """
    if n == 1:
        # Arity-1 element (1,) acts as the identity; sign = 1.
        return 1 if u == (1,) else 0

    perms = _extract_concatenated_permutations(u, n, d)
    if perms is None:
        return 0
    assert len(perms) == d + 1

    if d == 0:
        # Degree-0 surjection acting on S^0: BF sign is always +1.
        return 1

    sign_exp = d * n * (n - 1) // 2
    sign_exp += (d * (d - 1) // 2) * ((n - 2) * (n - 1) // 2)
    sign = -1 if sign_exp % 2 else 1
    for j in range(d):
        sign *= _permutation_sign(perms[j])

    return sign


class ReducedSphereCochains(CombinatorialFreeModule):
    r"""Reduced cochains of ``S^d`` as a rank-1 graded module.

    The reduced cochain complex `N^*(S^d; k)` is the one-dimensional
    `k`-module spanned by the fundamental class `g` in degree `d` (using
    the homological grading convention `\deg(g) = d`).  The boundary is
    zero.
    """

    def __init__(self, d: int, base_ring):
        assert d >= 0
        self._generator_name = f"ɑ{d}"
        super().__init__(
            base_ring,
            [self._generator_name],
            prefix=f"N*𝐒{d}",
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(self.prefix())
        self._generator = self(self._generator_name)
        self._sphere_dim = d
        self.connectivity = -d
        self.boundary = self.module_morphism(on_basis=lambda _: self.zero(), codomain=self)

    def sphere_dim(self) -> int:
        """Return the sphere dimension ``d``."""
        return self._sphere_dim

    def degree_on_basis(self, _) -> int:
        return -self._sphere_dim

    def generator(self):
        """Return the generator of the reduced cochains."""
        return self._generator

    def basis_iter(self, d: int):
        if d == -self._sphere_dim:
            yield self._generator

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    def basis_weight_iter(self, d: int, w: int):
        if w == 0:
            yield from self.basis_iter(d)

    @cached_method
    def graded_weighted_basis(self, d: int, w: int):
        return Family(self.basis_weight_iter(d, w))

    def _repr_term(self, element) -> str:
        return self._generator_name

    def _latex_term(self, element) -> str:
        return f"\\alpha^{{{self._sphere_dim}}}"

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


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

    def __init__(self, d: int, base_ring):
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
            if basis == self.module._generator_name:
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
