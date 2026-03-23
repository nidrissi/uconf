"""Regression tests for Surjection cooperad primitives.

The cooperad axioms tested here are:

* **Square-zero differential** – ``d² = 0`` on every basis element of
  ``SurjectionDual(n, QQ)``.

* **Sequential coassociativity** – for ``x ∈ S*(m+n+p-2)``, ``1 ≤ i ≤ m``,
  ``1 ≤ j ≤ n``::

      (id_m ⊗ Δ^{j;n,p}) ∘ Δ^{i;m,n+p-1}(x)
      = (Δ^{i;m,n} ⊗ id_p) ∘ Δ^{i+j-1;m+n-1,p}(x)

* **Parallel coassociativity** – for ``x ∈ S*(m+n+p-2)``, ``1 ≤ i < j ≤ m``::

      (Δ^{i;m,n} ⊗ id_p) ∘ Δ^{j+n-1;m+n-1,p}(x)
      = (-1)^{|b||c|} τ ∘ (Δ^{j;m,p} ⊗ id_n) ∘ Δ^{i;m+p-1,n}(x)

  where ``τ`` swaps the ``S*(n)`` and ``S*(p)`` factors and ``|b|``, ``|c|``
  are the degrees of the respective components.

* **Equivariance** – applying a block permutation to ``x`` is equivalent to
  applying the component permutations to each factor of the cocomposition.

* **Coderivation property** – ``Δ`` intertwines the differentials::

      Δ^{i;m,n}(d(x)) = (d ⊗ id)(Δ^{i;m,n}(x)) + (-1)^{|a|} (id ⊗ d)(Δ^{i;m,n}(x))

  where ``|a|`` is the degree of the left factor.
"""

from random import Random

import pytest
from sage.all import QQ

from uconf import Surjection, SurjectionDual


def _canonical(x):
    if hasattr(x, "tuple"):
        return tuple(int(v) for v in x.tuple())
    if isinstance(x, tuple):
        return tuple(_canonical(v) for v in x)
    return x


def _as_dict(x):
    return {_canonical(k): int(v) for k, v in x if int(v) != 0}


def _tensor_coeff(delta, left_basis, right_basis) -> int:
    d = _as_dict(delta)
    return d.get((_canonical(left_basis), _canonical(right_basis)), 0)


# ---------------------------------------------------------------------------
# Helper: represent a cocomposition as a flat {(left_key, right_key): coeff} dict
# ---------------------------------------------------------------------------


def _flat(delta) -> dict:
    """Return ``{(l_key, r_key): coeff}`` from a degree-2 tensor element."""
    return {(l, r): int(c) for (l, r), c in delta if int(c) != 0}


def _seq_lhs(x, i, j, m, n, p) -> dict:
    """``(id_m ⊗ Δ^{j;n,p}) ∘ Δ^{i;m,n+p-1}(x)`` as a flat triple dict."""
    SL = SurjectionDual
    base_ring = x.parent().base_ring()
    d1 = SL.infinitesimal_cocompose(x, i, m, n + p - 1)
    result: dict = {}
    for (a, d_key), c1 in d1:
        d_elem = SL(n + p - 1, base_ring)(d_key)
        for (b, c_key), c2 in SL.infinitesimal_cocompose(d_elem, j, n, p):
            key = (a, b, c_key)
            result[key] = result.get(key, 0) + int(c1) * int(c2)
    return {k: v for k, v in result.items() if v != 0}


def _seq_rhs(x, i, j, m, n, p) -> dict:
    """``(Δ^{i;m,n} ⊗ id_p) ∘ Δ^{i+j-1;m+n-1,p}(x)`` as a flat triple dict."""
    SL = SurjectionDual
    base_ring = x.parent().base_ring()
    d1 = SL.infinitesimal_cocompose(x, i + j - 1, m + n - 1, p)
    result: dict = {}
    for (e, c_key), c1 in d1:
        e_elem = SL(m + n - 1, base_ring)(e)
        for (a, b), c2 in SL.infinitesimal_cocompose(e_elem, i, m, n):
            key = (a, b, c_key)
            result[key] = result.get(key, 0) + int(c1) * int(c2)
    return {k: v for k, v in result.items() if v != 0}


def _par_lhs(x, i, j, m, n, p) -> dict:
    """``(Δ^{i;m,n} ⊗ id_p) ∘ Δ^{j+n-1;m+n-1,p}(x)`` as flat triple dict."""
    SL = SurjectionDual
    base_ring = x.parent().base_ring()
    d1 = SL.infinitesimal_cocompose(x, j + n - 1, m + n - 1, p)
    result: dict = {}
    for (e, c_key), c1 in d1:
        e_elem = SL(m + n - 1, base_ring)(e)
        for (a, b), c2 in SL.infinitesimal_cocompose(e_elem, i, m, n):
            key = (a, b, c_key)
            result[key] = result.get(key, 0) + int(c1) * int(c2)
    return {k: v for k, v in result.items() if v != 0}


def _par_rhs(x, i, j, m, n, p) -> dict:
    """``(-1)^{|b||c|} τ ∘ (Δ^{j;m,p} ⊗ id_n) ∘ Δ^{i;m+p-1,n}(x)`` as flat triple dict.

    ``τ`` swaps the S*(n) and S*(p) factors back to (a, b, c) order.
    """
    SL = SurjectionDual
    base_ring = x.parent().base_ring()
    Sn = SL(n, base_ring)
    Sp = SL(p, base_ring)
    d1 = SL.infinitesimal_cocompose(x, i, m + p - 1, n)
    result: dict = {}
    for (f, b), c1 in d1:
        f_elem = SL(m + p - 1, base_ring)(f)
        b_deg = Sn.degree_on_basis(b)
        for (a, c_key), c2 in SL.infinitesimal_cocompose(f_elem, j, m, p):
            c_deg = Sp.degree_on_basis(c_key)
            sign = (-1) ** (b_deg * c_deg)
            key = (a, b, c_key)
            result[key] = result.get(key, 0) + sign * int(c1) * int(c2)
    return {k: v for k, v in result.items() if v != 0}


def _coderivation_lhs(x, i, m, n) -> dict:
    """``Δ^{i;m,n}(d(x))`` as flat dict."""
    SL = SurjectionDual
    return _flat(SL.infinitesimal_cocompose(x.boundary(), i, m, n))


def _coderivation_rhs(x, i, m, n) -> dict:
    """``(d⊗id + (-1)^{|a|} id⊗d) Δ^{i;m,n}(x)`` as flat dict."""
    SL = SurjectionDual
    base_ring = x.parent().base_ring()
    Sl = SL(m, base_ring)
    Sr = SL(n, base_ring)
    result: dict = {}
    for (l_key, r_key), c in SL.infinitesimal_cocompose(x, i, m, n):
        sign = (-1) ** Sl.degree_on_basis(l_key)
        for dl_key, dc in Sl(l_key).boundary():
            key = (dl_key, r_key)
            result[key] = result.get(key, 0) + int(c) * int(dc)
        for dr_key, dc in Sr(r_key).boundary():
            key = (l_key, dr_key)
            result[key] = result.get(key, 0) + sign * int(c) * int(dc)
    return {k: v for k, v in result.items() if v != 0}


def _right_block_perm(tau_oneline: list[int], i: int, m: int, n: int) -> list[int]:
    """Build ``(id_m ×_i τ) ∈ S_{m+n-1}`` in one-line notation (1-indexed).

    Acts as the identity outside positions ``i, ..., i+n-1`` and applies τ
    on those positions by relabelling values ``i, ..., i+n-1`` according to τ.
    """
    total = m + n - 1
    result = list(range(1, total + 1))
    for k in range(i, i + n):
        result[k - 1] = i + tau_oneline[k - i] - 1
    return result


# ===========================================================================
# Original unit tests
# ===========================================================================


def test_surjection_counit_unit_and_reduced() -> None:
    s1 = SurjectionDual(1, QQ)
    unit = s1((1,))
    x = 3 * unit

    assert SurjectionDual.counit(unit) == 1
    assert unit.counit() == 1
    assert SurjectionDual.counit(x) == 3
    assert x.reduced() == s1.zero()


def test_surjection_counit_vanishes_outside_arity_one() -> None:
    x = SurjectionDual(2, QQ)((1, 2))
    assert SurjectionDual.counit(x) == 0
    assert x.reduced() == x


def test_infinitesimal_cocompose_transposes_compose_pairing() -> None:
    left = Surjection(2, QQ)((1, 2, 1))
    right = Surjection(2, QQ)((1, 2))
    i = 1

    composed = Surjection.compose(left, i, right)
    left_basis = next(iter(left.support()))
    right_basis = next(iter(right.support()))

    for u_basis, coeff in composed:
        u = SurjectionDual(3, QQ)(u_basis)
        delta = u.infinitesimal_cocompose(i=i, m=2, n=2)
        assert _tensor_coeff(delta, left_basis, right_basis) == int(coeff)


# ===========================================================================
# Square-zero differential:  d² = 0
# ===========================================================================


def test_differential_squared_zero() -> None:
    """d²(x) = 0 for every degree-d basis element of S*(n)."""
    rng = Random(20260317)
    SL = SurjectionDual
    for _ in range(10):
        n = rng.randint(2, 4)
        d = rng.randint(-2, 0)
        basis = list(SL(n, QQ).basis_iter(d))
        if not basis:
            continue
        elem = rng.choice(basis)

        d2 = elem.boundary().boundary()
        assert d2 == SL(n, QQ).zero(), f"d²({elem}) = {d2} ≠ 0 in S*({n}) degree {d}"


# ===========================================================================
# Sequential coassociativity
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,i,j,m,n,p",
    [
        # arity 4 = m+n+p-2, degree 0
        ((1, 2, 3, 4), 1, 1, 2, 2, 2),
        ((1, 2, 3, 4), 1, 2, 2, 2, 2),
        ((1, 2, 3, 4), 2, 1, 2, 2, 2),
        # arity 4, degree 1
        ((1, 2, 3, 1), 1, 1, 2, 2, 2),
        ((1, 2, 3, 1), 1, 2, 2, 2, 2),
        ((2, 3, 1, 2), 2, 1, 2, 2, 2),
        # arity 5 = 3+2+2-2, m=3,n=2,p=2
        ((1, 2, 3, 4, 5), 1, 1, 3, 2, 2),
        ((1, 2, 3, 4, 5), 2, 1, 3, 2, 2),
        ((1, 2, 3, 4, 5), 3, 2, 3, 2, 2),
        # arity 5 = 2+3+2-2, m=2,n=3,p=2
        ((1, 2, 3, 4, 5), 1, 1, 2, 3, 2),
        ((1, 2, 3, 4, 5), 1, 2, 2, 3, 2),
        ((1, 2, 3, 4, 5), 2, 3, 2, 3, 2),
        # arity 5 degree 1
        ((1, 2, 3, 4, 1), 1, 1, 3, 2, 2),
        ((1, 2, 3, 4, 1), 2, 1, 3, 2, 2),
        ((2, 3, 1, 4, 2), 1, 2, 2, 3, 2),
        # arity 6 = 3+2+3-2, m=3,n=2,p=3, degree 0
        ((1, 2, 3, 4, 5, 6), 1, 1, 3, 2, 3),
        ((1, 2, 3, 4, 5, 6), 2, 2, 3, 2, 3),
        ((1, 2, 3, 4, 5, 6), 3, 1, 3, 2, 3),
    ],
)
def test_sequential_coassociativity(x_tuple: tuple, i: int, j: int, m: int, n: int, p: int) -> None:
    """(id⊗Δ^{j;n,p})∘Δ^{i;m,n+p-1} = (Δ^{i;m,n}⊗id)∘Δ^{i+j-1;m+n-1,p}."""
    x = SurjectionDual(m + n + p - 2, QQ)(x_tuple)
    lhs = _seq_lhs(x, i, j, m, n, p)
    rhs = _seq_rhs(x, i, j, m, n, p)
    assert lhs == rhs, (
        f"Sequential coassociativity failed for x={x_tuple}, "
        f"i={i}, j={j}, m={m}, n={n}, p={p}\nLHS={lhs}\nRHS={rhs}"
    )


# ===========================================================================
# Parallel coassociativity
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,i,j,m,n,p",
    [
        # m=2, n=p=2, total arity 4, i=1 < j=2 ≤ m=2
        ((1, 2, 3, 4), 1, 2, 2, 2, 2),
        ((1, 2, 3, 1), 1, 2, 2, 2, 2),
        ((2, 1, 3, 2), 1, 2, 2, 2, 2),
        # m=3, n=p=2, total arity 5
        ((1, 2, 3, 4, 5), 1, 2, 3, 2, 2),
        ((1, 2, 3, 4, 5), 1, 3, 3, 2, 2),
        ((1, 2, 3, 4, 5), 2, 3, 3, 2, 2),
        ((1, 2, 3, 4, 1), 1, 2, 3, 2, 2),
        ((1, 2, 3, 4, 1), 2, 3, 3, 2, 2),
        # m=4, n=p=2, total arity 6, all i<j pairs
        ((1, 2, 3, 4, 5, 6), 1, 2, 4, 2, 2),
        ((1, 2, 3, 4, 5, 6), 1, 3, 4, 2, 2),
        ((1, 2, 3, 4, 5, 6), 2, 4, 4, 2, 2),
        ((1, 2, 3, 4, 5, 6), 3, 4, 4, 2, 2),
        # m=3, n=2, p=3, total arity 6
        ((1, 2, 3, 4, 5, 6), 1, 2, 3, 2, 3),
        ((1, 2, 3, 4, 5, 6), 1, 3, 3, 2, 3),
        ((1, 2, 3, 4, 5, 6), 2, 3, 3, 2, 3),
    ],
)
def test_parallel_coassociativity(x_tuple: tuple, i: int, j: int, m: int, n: int, p: int) -> None:
    """(Δ^{i;m,n}⊗id)∘Δ^{j+n-1;m+n-1,p} = (-1)^{|b||c|} τ∘(Δ^{j;m,p}⊗id)∘Δ^{i;m+p-1,n}."""
    x = SurjectionDual(m + n + p - 2, QQ)(x_tuple)
    lhs = _par_lhs(x, i, j, m, n, p)
    rhs = _par_rhs(x, i, j, m, n, p)
    assert lhs == rhs, (
        f"Parallel coassociativity failed for x={x_tuple}, "
        f"i={i}, j={j}, m={m}, n={n}, p={p}\nLHS={lhs}\nRHS={rhs}"
    )


# ===========================================================================
# Equivariance
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,tau_oneline,i,m,n",
    [
        # m=n=2, i=1, τ=(2,1) ∈ S_2 swaps the two right inputs
        ((1, 2, 3), [2, 1], 1, 2, 2),
        ((1, 2, 1), [2, 1], 1, 2, 2),
        ((2, 1, 2), [2, 1], 1, 2, 2),
        # m=2, n=3, i=1, τ=(2,3,1) ∈ S_3
        ((1, 2, 3, 4), [2, 3, 1], 1, 2, 3),
        ((1, 2, 3, 1), [2, 3, 1], 1, 2, 3),
        # m=3, n=2, i=2, τ=(2,1) ∈ S_2 — right block occupies values 2,3 of S*(4)
        ((1, 2, 3, 4), [2, 1], 2, 3, 2),
        ((2, 1, 3, 2), [2, 1], 2, 3, 2),
        # m=3, n=2, i=3, τ=(2,1) ∈ S_2 — right block at positions 3,4
        ((1, 2, 3, 4), [2, 1], 3, 3, 2),
        ((1, 2, 3, 1), [2, 1], 3, 3, 2),
        # m=2, n=3, i=1, τ=(3,1,2) ∈ S_3
        ((1, 2, 3, 4), [3, 1, 2], 1, 2, 3),
        ((2, 1, 3, 4), [3, 1, 2], 1, 2, 3),
        # m=3, n=3, i=1, τ=(2,1,3) ∈ S_3
        ((1, 2, 3, 4, 5), [2, 1, 3], 1, 3, 3),
        ((1, 2, 3, 4, 1), [2, 1, 3], 1, 3, 3),
    ],
)
def test_equivariance_right_block(
    x_tuple: tuple, tau_oneline: list[int], i: int, m: int, n: int
) -> None:
    """Δ^{i;m,n}(x·(id_m ×_i τ)) = (id ⊗ τ)·Δ^{i;m,n}(x).

    Acting on x by the block permutation that applies τ to the n inputs at
    positions i..i+n-1 is equivalent to applying τ to the right factor of
    the cocomposition.
    """
    SL = SurjectionDual
    x = SL(m + n - 1, QQ)(x_tuple)
    rho = _right_block_perm(tau_oneline, i, m, n)

    # LHS: Δ(x · rho)
    lhs = _flat(SL.infinitesimal_cocompose(x.permute(rho), i, m, n))

    # RHS: apply τ to the right factor of Δ(x)
    rhs: dict = {}
    for (l_key, r_key), coeff in SL.infinitesimal_cocompose(x, i, m, n):
        new_r = tuple(tau_oneline[v - 1] for v in r_key)
        key = (l_key, new_r)
        rhs[key] = rhs.get(key, 0) + int(coeff)
    rhs = {k: v for k, v in rhs.items() if v != 0}

    assert lhs == rhs, (
        f"Equivariance failed for x={x_tuple}, τ={tau_oneline}, "
        f"i={i}, m={m}, n={n}\nLHS={lhs}\nRHS={rhs}"
    )


# ===========================================================================
# Coderivation property
# ===========================================================================


@pytest.mark.parametrize(
    "x_tuple,i,m,n",
    [
        # S*(3), degree 1
        ((1, 2, 1), 1, 2, 2),
        ((2, 1, 2), 1, 2, 2),
        ((1, 2, 1), 2, 2, 2),
        # S*(4), degree 1
        ((1, 2, 3, 1), 1, 2, 3),
        ((1, 2, 3, 1), 2, 2, 3),
        ((1, 2, 3, 1), 1, 3, 2),
        ((1, 2, 3, 1), 2, 3, 2),
        ((1, 2, 3, 1), 3, 3, 2),
        ((2, 3, 1, 2), 1, 2, 3),
        # S*(4), degree 2
        ((1, 2, 1, 2), 1, 2, 3),
        ((1, 2, 1, 2), 2, 2, 3),
        ((1, 2, 1, 2), 1, 3, 2),
        # S*(5), degree 1
        ((1, 2, 3, 4, 1), 1, 3, 3),
        ((1, 2, 3, 4, 1), 2, 3, 3),
        ((1, 2, 3, 4, 1), 3, 3, 3),
        ((1, 2, 3, 4, 1), 1, 2, 4),
        ((1, 2, 3, 4, 1), 2, 2, 4),
        # S*(5), degree 2
        ((1, 2, 3, 1, 2), 1, 2, 4),
        ((1, 2, 3, 1, 2), 2, 3, 3),
        ((1, 2, 3, 1, 2), 1, 4, 2),
        # S*(6), degree 1
        ((1, 2, 3, 4, 5, 1), 1, 3, 4),
        ((1, 2, 3, 4, 5, 1), 3, 3, 4),
        ((1, 2, 3, 4, 5, 1), 1, 4, 3),
    ],
)
def test_coderivation_property(x_tuple: tuple, i: int, m: int, n: int) -> None:
    """Δ(d(x)) = (d⊗id + (-1)^{|a|} id⊗d) Δ(x)."""
    x = SurjectionDual(m + n - 1, QQ)(x_tuple)
    lhs = _coderivation_lhs(x, i, m, n)
    rhs = _coderivation_rhs(x, i, m, n)
    assert lhs == rhs, (
        f"Coderivation failed for x={x_tuple}, i={i}, m={m}, n={n}\nLHS={lhs}\nRHS={rhs}"
    )
