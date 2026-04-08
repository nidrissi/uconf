# Performance Optimizations

This document summarises the performance optimizations applied to `uconf`,
both historical and current. Measurements reference the benchmark in
`benchmark.py`: computing `compute_chain_complex` for the euclidean
unordered configuration model on R² over GF(2) at weight 4.

---

## Existing optimizations

### 1. `RootedTree` with eagerly cached structural properties

**Files:** `src/uconf/core/trees.py`

Replaced the legacy nested-tuple tree representation with an immutable,
hashable `RootedTree` class that computes `weight`, `leaves`, `min_leaf`,
`tree_arity`, and `hash` at construction time. All structural queries are
now O(1) instead of O(tree_size).

This eliminates repeated O(v) traversals in hot paths such as
`to_shuffle_tree_bar/cobar`, `validate_tree`, and `_d2_on_basis`.

See [tree-redesign.md](tree-redesign.md) for the full design rationale.

### 2. Planar coinvariant enumeration (~2.8× speedup)

**Files:** `src/uconf/algebraic/cofree_coalgebra.py`,
`src/uconf/algebraic/tree_module.py`

Instead of explicitly constructing full S_n orbit sums for the cofree
coalgebra basis, enumerate planar representatives directly and canonicalize
non-planar boundary output back to planar keys. Uses the isomorphism
`(P_pl ⊗ k[S_n]) ⊗_{S_n} M^{⊗n} ≅ P_pl ⊗ M^{⊗n}`.

Observed speedup: **~2.8×** (226s → 80s on heavy layers). Validated
`d² = 0` over QQ and GF(2).

See [tree-redesign.md](tree-redesign.md#status-update-2026-04-planar-coinvariant-enumeration).

### 3. Connectivity short-circuit in `compute_chain_complex`

**File:** `src/uconf/homology.py`

If a degree is below the module's connectivity, the basis is trivially
empty. Skips basis enumeration and boundary matrix assembly entirely for
such degrees.

### 4. Sparse matrix assembly

**File:** `src/uconf/homology.py`

`_boundary_matrix` uses SageMath's sparse matrix format by default, which
is faster and more memory-efficient for the typically sparse differentials.

### 5. `@cached_method` on boundary/differential methods

**Files:** `bar_construction.py`, `cobar_construction.py`,
`bar_algebra.py`, `cobar_coalgebra.py`, `tree_module.py`,
`cofree_coalgebra.py`

The methods `_boundary_on_basis`, `_d1_on_basis`, `_d2_on_basis`,
`_dalpha_on_basis`, and `_twisted_boundary_on_basis` are all cached
via SageMath's `@cached_method`. Since trees are immutable and hashable,
cache lookups are efficient and results are never recomputed.

### 6. Pre-cached permutation lists

**File:** `src/uconf/morphisms/e_comodule_morphism.py`

`_non_identity_perms(n)` is decorated with `@cached_function`,
materialising the non-identity elements of S_n exactly once per arity for
the lifetime of the process. This avoids repeated GAP bridge calls.

### 7. Validated-tree fast paths (`_from_validated_tree`)

**Files:** `bar_construction.py`, `cobar_construction.py`

Internal boundary computations produce trees from known-valid inputs.
`_from_validated_tree` skips `_validate_basis_key` (and its recursive
`validate_tree` traversal) while still normalising to shuffle form.

---

## Optimizations applied 2026-04-08

### Benchmark baseline

| Metric | Before | After |
|--------|--------|-------|
| Wall-clock time | 29.14 s | 11.97 s |
| **Speedup** | — | **2.43×** |
| `d_sigma` calls | 132,567 | 5,829 |
| `d_sigma` cumtime | 9.45 s | 0.70 s |
| `validate_tree` calls | 113,556 | 78,269 |
| `validate_tree` cumtime | 3.88 s | 2.83 s |
| `RootedTree.__init__` calls | 187,888 | 150,817 |
| `e_comodule` pipeline cumtime | 13.57 s | 3.75 s |
| Total function calls | 30.5 M | 18.3 M |

### 8. Batched `d_sigma_decompose` (dominant optimisation)

**Files:** `src/uconf/core/quasi_planar.py`,
`src/uconf/morphisms/e_comodule_morphism.py`

**Problem.** The recursive E-comodule computation (`_nu_on_planar.recurse`)
called `d_sigma(x, σ)` once per non-identity permutation σ ∈ S_n. Each
call independently computed `boundary(x)` and `planarize(∂x)`, then
scanned the result for the single σ-component. For arity n = 4;
`|S_4| − 1 = 23` calls per element, totalling 132 K calls and 9.45 s.

**Fix.** Added `d_sigma_decompose(x)` to `QuasiPlanarMixin`, which
computes `boundary` + `planarize` **once** and returns a dictionary
`{σ: d_σ(x)}` of all non-zero components. Modified `recurse` to call
`d_sigma_decompose` once per element and iterate over the (sparse) result.

**Impact.** 132 K → 5.8 K calls (22.7× reduction). Cumulative time
9.45 s → 0.70 s (**13.5× speedup** on this path). Overall benchmark
improvement of roughly 10 s.

### 9. Memoized `subtree_degree` / `subtree_degree_cobar`

**File:** `src/uconf/core/trees.py`

Both functions recursively compute the total shifted bar/cobar degree of
a subtree. Since `RootedTree` instances are immutable and hashable, the
results are now cached via `@lru_cache(maxsize=None)` keyed on
`(tree, operad_cls, base_ring)`. Called during shuffle tree normalisation
(`to_shuffle_tree_bar/cobar`) and sign computation.

### 10. Bypass validation in `_extend_tree`

**File:** `src/uconf/morphisms/e_comodule_morphism.py`

In `_extend_tree`, cobar elements are constructed from known-valid
cooperad keys (produced by `e_comodule_on_generator`). Changed
`cobar_k(cobar_tree)` to `cobar_k._from_validated_tree(cobar_tree)` and
`target_k((be_key, cobar_key))` to `target_k.term((be_key, cobar_key))`,
skipping redundant validation and shuffle normalisation for single-vertex
corollas.

**Impact.** Reduced `validate_tree` calls by ~31 % (113 K → 78 K) and
`hadamard_operad._validate_basis_key` calls by ~34 %.

---

## Remaining bottlenecks (for future work)

After the 2026-04-08 optimisations, the top cumulative-time functions are:

| Function | Cumtime | Notes |
|----------|---------|-------|
| `hadamard_algebra._act_impl` | 9.5 s | Cartesian-product term expansion |
| `free_algebra._act_impl` | 6.3 s | Operadic algebra action |
| `free_algebra._normalized_corolla_sum` | 4.1 s | Normalisation after composition |
| `validate_tree` / `validate_vertex` | 2.8 s | Still ~78 K calls from `_element_constructor_` |
| `_planarize_on_basis` (bar + cobar) | 5.4 s | Planar decomposition |
| `cobar_construction.compose` | 2.7 s | Tree-based operadic composition |

Potential future optimisations:

- **Skip validation in more internal paths:** Many `_element_constructor_`
  calls from SageMath's `sum_of_terms` / `_from_dict` could use
  `_from_validated_tree` when the caller guarantees validity.
- **Cache `Lie._permute_on_basis`:** Add `@cached_method` to match
  `BarConstruction._permute_on_basis`.
- **GF(2) sign short-circuit:** Over characteristic 2, skip Koszul sign
  computation entirely (`sign_from_exponent` always returns 1).
- **Batch boundary + planarize in `_planarize_on_basis`:** Similar
  batching strategy as `d_sigma_decompose`.
