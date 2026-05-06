# Performance Optimizations

This document summarises the performance optimizations applied to `uconf`,
both historical and current. Measurements reference the benchmark in
`benchmark.py`: computing `compute_chain_complex` for the euclidean
unordered configuration model on R² over GF(2) at weight 3.

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
results are now cached via `@cached_function` keyed on
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

## Optimizations applied 2026-04-09

### Benchmark baseline (weight 3)

| Metric | Before | After |
|--------|--------|-------|
| Wall-clock time | ~10 s | ~4.3 s |
| **Speedup** | — | **~2.3×** |
| Total function calls | ~18.3 M | ~8.6 M |

### 11. Cached table reduction per basis key

**File:** `src/uconf/models/__init__.py`

Introduced `_compute_table_reduction_cached` decorated with
`@cached_function` keyed on `(arity, base_ring, basis_key)`.
Previously table reduction was recomputed each time a BE basis element
appeared in a different context; this eliminates ~8× redundant
recomputation.

### 12. Bypass Sage `morphism.__call__` overhead

**Files:** `src/uconf/homology.py`, `src/uconf/algebraic/free_algebra.py`,
`src/uconf/algebraic/cofree_coalgebra.py`,
`src/uconf/algebraic/hadamard_algebra.py`,
`src/uconf/core/quasi_planar.py`

Each `morphism(term(key))` call adds ~2 µs overhead from
`linear_combination` / generator dispatch. For 90 K+ calls this
dominates.

Extracted `get_on_basis()` helper in `core/signs.py` to uniformly access
the underlying `on_basis` callable. Applied in:

- `_boundary_matrix`: call `on_basis(key)` directly instead of
  `boundary(elem)`
- `_normalized_corolla_sum`: call `_planarize_on_basis(key)` directly
  instead of `planarize(term(key))`
- `d_sigma_decompose`: bypass both boundary and planarize wrappers
- `free_algebra._boundary_on_basis`: call inner-module boundary directly

### 13. Dict accumulation over element construction

**Files:** `src/uconf/algebraic/hadamard_algebra.py`,
`src/uconf/morphisms/e_comodule_morphism.py`,
`src/uconf/algebraic/configuration.py`,
`src/uconf/algebraic/free_algebra.py`,
`src/uconf/algebraic/cofree_coalgebra.py`

Replaced the pattern `result += coeff * module.term(key)` (which creates
an intermediate Sage element per term) with `{key: coeff}` dict
accumulation followed by a single `sum_of_terms()` call. Eliminates
thousands of intermediate Sage element constructions per boundary
evaluation.

### 14. Cache morphism result in `PullbackAlgebra.act`

**File:** `src/uconf/algebraic/pullback_algebra.py`

The pullback algebra action `γ^Q(f(p); a_1, …, a_n)` calls
`morphism(p_element)` for each `p`. The morphism result depends only
on `p`, not on the algebra inputs `a_i`. Added a per-instance dict
cache keyed on `tuple(p_element)` to avoid redundant morphism
evaluations.

### 15. Bypass validation for known-valid keys

**Files:** `src/uconf/constructions/cobar_construction.py`,
`src/uconf/algebraic/hadamard_algebra.py`,
`src/uconf/models/surjection.py`

- Use `_from_validated_tree` in cobar `compose` for internally-grafted
  trees.
- Use `self.module.term(key)` instead of `self.module(key)` in
  `hadamard_algebra._act_impl` for keys that are already validated.
- Optimise `surjection._validate_basis_key` to a single-pass combined
  bounds + degeneracy check.

### 16. Direct `_compute_table_reduction_cached` call

**File:** `src/uconf/algebraic/configuration.py`

`configuration._on_element` now calls the `@cached_function` directly
instead of constructing a BE element and invoking
`element.table_reduction()`, bypassing element construction and morphism
dispatch.

---

## Remaining bottlenecks (for future work)

After the 2026-04-09 optimisations, the top cumulative-time functions are
(weight-3 benchmark, first run):

| Function | Cumtime | Notes |
|----------|---------|-------|
| `e_comodule` pipeline | 3.6 s | 45 % — recursive cooperad coaction |
| `hadamard_algebra._act_impl` | 2.6 s | Cartesian-product term expansion |
| `free_algebra._act_impl` | 1.8 s | Operadic algebra action |
| `_compute_table_reduction_cached` | 0.85 s | Partition enumeration (one-time) |
| `sum_of_terms` | 1.3 s | 83 K calls — Sage element construction |
| `_boundary_matrix` (total) | 8.0 s | Full boundary loop over 650 basis elements |

### Evaluated optimisation leads

The following were listed as potential future optimisations (2026-04-08)
and have been evaluated against the current profile:

- **Cache `Lie._permute_on_basis`:** ✅ Already implemented (has
  `@cached_method` since the 2026-04-08 round).
- **GF(2) sign short-circuit:** ❌ Not worthwhile. Profiling shows
  `sign_from_exponent` and `koszul_sign_of_permutation` together total
  only ~30 ms (0.04 s), less than 1 % of wall time. The refactoring cost
  (threading base-ring characteristic through every call site) outweighs
  the benefit.
- **Skip validation in more internal paths:** ✅ Partially done (see §15).
  Remaining `validate_tree`/`validate_vertex` calls (~84 K, 0.47 s)
  come from `_from_validated_tree` → `_normalize_to_shuffle` → shuffle
  tree normalisation, which still validates vertex ordering. These are
  structurally necessary.
- **Batch boundary + planarize in `_planarize_on_basis`:** ✅ Already
  achieved via `d_sigma_decompose` (§8) and recursive `_planarize_subtree`.

### New leads

- **E-comodule morphism caching or algebraic reformulation:** The
  recursive `_nu_on_planar.recurse` function (1.1 s, 9.7 K calls)
  dominates the pipeline. Caching intermediate results keyed on
  `(current_d_elem, len(sigma_bar))` could help if many branches
  converge to the same intermediate element. Alternatively, an
  algebraic reformulation that avoids the recursion entirely (e.g.
  computing the acyclic-models map via iterated tensor products)
  could yield larger gains.
- **Reduce `sum_of_terms` overhead:** With 83 K calls at 1.3 s, the
  Sage `CombinatorialFreeModule.sum_of_terms` method is a significant
  overhead. For internal constructions that produce validated keys,
  directly building the `_from_dict` representation could bypass the
  `sum_of_terms` → `_from_dict` → `__classcall_private__` chain.
- **Table reduction algorithm:** The current `term_generator` (0.8 s)
  enumerates all permutations of integer partitions via
  `set(permutations(pi_ord))`. For larger arities, a direct algorithm
  that avoids redundant partition permutations could reduce this
  one-time cost.
- **Parallel boundary matrix assembly:** Since boundary evaluations for
  different basis elements are independent, the `_boundary_matrix` loop
  could be parallelised across multiple processes, though Sage's GIL
  constraints limit the benefit of threading.

---

## Optimizations applied 2026-04-27

### 17. Direct `on_basis` path in `_boundary_matrix`

**File:** `src/uconf/homology.py`

`compute_chain_complex` used to assemble each differential column by calling
`module.boundary(elem)` on a one-term basis element. Even when the module
exposed an `on_basis` callback, Sage still paid the full
`morphism.__call__ -> linear_combination` overhead for every column.

`_boundary_matrix` now detects the underlying `on_basis` function with
`get_on_basis()` and calls it directly on the source key. This preserves the
existing fallback path for generic modules, while avoiding thousands of tiny
single-term morphism evaluations during chain-complex assembly.

To keep compatibility with modules that normalize keys only at construction
time, `_boundary_matrix` still falls back to `module._normalize_key(...)`
when a returned key is not already present in the target basis.

### Benchmark impact

On the weight-3 Euclidean configuration benchmark (`GF(2)`, `dim=2`):

| Metric | Before | After |
|--------|--------|-------|
| `benchmark_detailed.py` full chain complex | 7.79 s | 4.29 s |
| `benchmark.py` (`cProfile`) total time | 15.80 s | 6.79 s |

After this change, the dominant remaining cost is no longer chain-complex
assembly overhead but the recursive Le Grignou-Roca i Lucio E-comodule map
(`src/uconf/morphisms/e_comodule_morphism.py`) together with the `d_alpha`
twisting-differential pipeline.

### 18. Dynamic-programming fast paths in the Le Grignou–Roca i Lucio comodule map

**File:** `src/uconf/morphisms/e_comodule_morphism.py`

The recursive cooperad-level map still spent most of its time in two kinds
of repeated work:

1. recomputing the cumulative permutation product
   `sigma_k * ... * sigma_1` from scratch at every recursion depth, and
2. rebuilding the same generator/subtree images many times while traversing
   cobar trees with repeated decorations.

This was addressed in three places:

- `_nu_on_planar.recurse` now threads the cumulative permutation product
  through the recursion instead of recomputing it from `sigma_bar`.
- The equivariant cooperad action uses direct `_permute_on_basis` fast paths
  when available, avoiding repeated element-wrapper `permute()` overhead.
- `make_e_comodule_morphism` memoizes both generator root images and full
  subtree images, turning the tree extension step into a small dynamic
  program over repeated decorations/subtrees.

### Benchmark impact

On the arity-3 comodule sub-benchmark in `benchmark_detailed.py`, the
Le Grignou–Roca i Lucio map dropped from about **6.2 s** before this round
to about **1–3 s** after it, depending on cache warmth and process startup.

In an isolated `cProfile` run over all arity-3 basis elements:

| Metric | Before | After |
|--------|--------|-------|
| Total time | 5.05 s | 2.43 s |
| `_extend_tree` cumtime | 4.59 s | 2.18 s |
| `e_comodule_on_generator` cumtime | 3.76 s | 1.76 s |
| `_nu_on_planar` cumtime | 1.40 s | 0.72 s |

---

## Optimizations applied 2026-04-30

### 19. Basis-level action caches in free/Hadamard algebras

**Files:** `src/uconf/algebraic/free_algebra.py`,
`src/uconf/algebraic/hadamard_algebra.py`

The bar differential repeatedly applies algebra actions to the **same basis
inputs** with different scalar coefficients.  Previously both
`FreeOperadAlgebra._act_impl` and `HadamardTensorAlgebra._act_impl`
recomputed the full substitution / graded-interchange / normalization pipeline
for every call, even when the underlying basis tuple had already appeared in an
earlier boundary term.

This round adds cached basis-level helpers:

- `FreeOperadAlgebra._act_on_basis_tuple(q_key, input_keys)`
- `HadamardTensorAlgebra._act_on_basis_tuple(had_basis, tensor_basis_tuple)`

The public `_act_impl` methods now only expand coefficients and reuse the
cached basis result.  This pushes repeated operadic substitution, normalization,
and tensor-product output construction behind `@cached_method`.

### 20. Cached bar/E-comodule intermediates and cheaper normalization

**Files:** `src/uconf/constructions/bar_algebra.py`,
`src/uconf/morphisms/e_comodule_morphism.py`,
`src/uconf/algebraic/cofree_coalgebra.py`,
`src/uconf/homology.py`

Several remaining hotspots were still rebuilding the same intermediate objects:

- `BarAlgebraModule._dalpha_*` now caches `α(c_R)` on basis keys and the
  normalized left-corolla outputs `(c_L, new_m)`.
- `_nu_on_planar` now memoizes repeated `d_sigma_decompose` calls, repeated
  cooperad-factor permutations, and repeated `rho(sigma_bar)` evaluations
  inside one recursion tree.
- `CofreeCoalgebraModule._normalized_corolla_sum` and `_boundary_on_basis`
  now accumulate directly into dictionaries instead of repeatedly doing
  `result += ...`.
- `_boundary_matrix` now caches normalized target-row matches for repeated raw
  boundary keys, reducing duplicate `_normalize_key(...)` work during matrix
  assembly.

### 21. Cold vs warm benchmark split

**Files:** `benchmark.py`, `benchmark_detailed.py`

The benchmark scripts now run **cold** and **warm** passes separately in the
same process so that one-time cache fill can be compared against steady-state
costs.

On the weight-3 Euclidean configuration benchmark (`GF(2)`, `dim=2`):

| Metric | Cold | Warm |
|--------|------|------|
| `benchmark_detailed.py` full chain complex | 19.58 s | 0.056 s |
| `benchmark_detailed.py` arity-3 E-comodule sub-benchmark | 5.22 s | 1.67 s |
| `benchmark_detailed.py` basis enumeration | 0.407 s | ~0.000 s |
| `benchmark.py` wall time (`n_jobs=8`) | 20.97 s | 18.98 s |

The serial detailed benchmark now makes the cache split explicit:

- **One-time setup + cache fill:** ~19.60 s
- **Steady-state full chain complex:** ~0.056 s

The parallel profiled benchmark remains much slower on the warm pass because
forked workers do not reuse the main process's cached `@cached_method` state;
the wall time is now dominated by process/lock overhead, while the merged
worker profiles still identify the same mathematical hot paths.

### Current bottleneck split after this round

From `benchmark_profile.txt` (merged worker profiles; cumulative times sum
across workers and therefore exceed wall clock):

- `bar_algebra._dalpha_all_splits`: **118.9 s** cumulative
- `configuration._on_element`: **88.4 s**
- `e_comodule_on_generator`: **70.7 s**
- `e_comodule._on_element`: **74.2 s**
- `hadamard_algebra._act_impl`: **28.6 s**
- `hadamard_algebra._act_on_basis_tuple`: **28.0 s**
- `free_algebra._act_impl`: **24.4 s**
- `free_algebra._act_on_basis_tuple`: **23.1 s**
- `quasi_planar.d_sigma_decompose`: **18.8 s**
- `cofree_coalgebra._boundary_on_basis`: **13.0 s**
- `cofree_coalgebra._normalized_corolla_sum`: **5.3 s**

So the next work should stay focused on the `d_alpha` / comodule / algebra
action stack, not on table reduction or basis enumeration.

---

## Optimizations applied 2026-05-06

### Benchmark baseline (weight 3, deg_max 5, serial)

| Metric | Before | After |
|--------|--------|-------|
| Wall-clock time | 27.98 s | 11.52 s |
| **Speedup** | — | **2.43×** |
| `term_generator` tottime | 16.50 s | 0.17 s |
| `_compute_table_reduction_cached` cumtime | 16.64 s | 0.49 s |
| `sum_of_terms` cumtime | 16.78 s | 0.61 s |

### 22. Pure-Python `_compute_table_reduction_cached` rewrite

**File:** `src/uconf/models/__init__.py`

**Problem.** The `term_generator` inner loop inside `_compute_table_reduction_cached`
used `Partitions(d+n, length=d+1)` + `set(permutations(pi_ord))` to enumerate
all ordered compositions of `d+n` into `d+1` positive parts. For a partition
like `[3,1,1,1,1]` this generated 720 permutation candidates and deduped to 5
unique tuples; the set/permutation overhead accumulated to **16.5 s tottime**
(60 % of wall time) across 3 666 cache misses.

Three issues were fixed:

1. **Composition enumeration via `itertools.combinations`** (1a): replaced
   the Sage `Partitions` + `permutations` + `set` triple with a pure-Python
   `_compositions(m, k)` generator that places `k-1` dividers in `m-1` slots.
   Yields each of the `C(m-1, k-1)` compositions exactly once with no
   deduplication. On the heaviest case (d=5, n=3) this is 19× faster in
   isolation; `sage.combinat.Compositions` was tried first but found to be
   36× *slower* than the current code due to Sage type-system overhead.

2. **Hoist `.tuple()` calls** (1b): precomputed
   `basis_tuples = tuple(s.tuple() for s in basis_element)` once per cached
   call instead of calling `.tuple()` inside both loop levels.

3. **Bitmask for `removed`** (1c): replaced the `removed: list` (O(n)
   membership) with a small-int bitmask (O(1) bitwise AND). Arities in this
   benchmark are ≤ 5 so a Python `int` suffices.

**Impact.** `term_generator` tottime: 16.5 s → 0.17 s (**97× on this
function**). Overall benchmark: 27.98 s → 11.52 s (**2.43× wall-clock**).
Table reduction is no longer in the top 10 by cumtime.

### Remaining bottlenecks after this round

From `benchmark_profile_2026-05-06_17-01-38_2_3_5_1.txt` (serial,
`-d2 -w3 -m5`):

| Function | cumtime |
|----------|---------|
| `bar_algebra._dalpha_all_splits` | 8.15 s |
| `pullback_algebra._act_on_basis_inputs` | 7.93 s |
| `configuration._on_element` | 4.77 s |
| `e_comodule_morphism._root_image_for_generator` | 4.16 s |
| `e_comodule_morphism._extend_tree` | 4.18 s |
| `hadamard_algebra._act_on_basis_tuple` | 3.10 s |
| `e_comodule_morphism._nu_on_planar` | 3.05 s |
| `e_comodule_morphism.recurse` | 2.85 s |
| `free_algebra._act_on_basis_tuple` | 2.15 s |

The dominant remaining cost is the `d_alpha` / E-comodule / algebra-action
stack — the same cluster identified after the 2026-04-30 round but now
exposed without the table-reduction overhead masking it.
