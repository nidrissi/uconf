# Tree Data Structure Redesign

## Problem

The current nested-tuple tree representation requires full recursive traversal
for every structural query (weight, arity, leaves, min_leaf, subtree_degree).
Since these queries are called repeatedly in performance-critical paths—especially
the bar/cobar differentials `_d2_on_basis`—this causes O(v²) or worse overhead
per differential computation.

## Current Representation

Trees are encoded as nested Python tuples:

- **Leaf**: `int` in `{1, ..., n}`
- **Internal vertex**: `(decoration, child_1, ..., child_k)` where
  `decoration` is an operad/cooperad basis key (a tuple) and each `child_j`
  is a leaf or subtree.

Example: `((1, 2), ((1,), 1, 2), 3)` — arity 3, weight 2.

### Bottlenecks

1. **`subtree_degree` recomputation in `_d2_on_basis`**: For each internal edge,
   the degrees of all sibling subtrees are recomputed from scratch. This is
   O(v² × lookup) per tree.

2. **`min_leaf` recomputation in shuffle normalization**: Called O(k) times per
   vertex during `to_shuffle_tree_bar/cobar`, each O(n) traversal.

3. **`leaves()` recomputation**: O(n) set union called in `after_cobar_deg`,
   `tree_arity`, `validate_tree`, and others.

4. **`weight()` recomputation**: O(v) recursive count called in enumeration.

5. **Full recursive copy on every structural operation**: `relabel_leaves`,
   `graft`, `contract_edge`, `expand_vertex` all rebuild the entire tree.

## New Design: `RootedTree` Class

### Core Idea

Replace nested tuples with an immutable `RootedTree` class that **eagerly
computes and caches structural properties** at construction time. Since every
tree operation already rebuilds the tree from scratch anyway, the per-node
overhead of caching is amortized over the many subsequent queries.

### Invariants

- Leaves remain as plain `int` values (no wrapping).
- `RootedTree` is immutable and hashable — works as SageMath `@cached_method`
  arguments and `CombinatorialFreeModule` basis keys.
- Identity comparison (`is`) continues to work for DFS-extracted vertices.

### Cached Properties (computed at construction, O(1) access)

| Property | Type | Old cost | New cost |
|----------|------|----------|----------|
| `weight` | `int` | O(v) | O(1) |
| `tree_arity` | `int` | O(n) | O(1) |
| `leaves` | `frozenset[int]` | O(n) | O(1) |
| `min_leaf` | `int` | O(n) | O(1) |
| `arity` (vertex) | `int` | O(1) | O(1) |
| `decoration` | `tuple` | O(1) | O(1) |
| `children` | `tuple` | O(1) | O(1) |

### Construction Cost

Each `RootedTree(decoration, children)` call computes cached properties by
merging child properties in O(k) time (where k = number of children).
Since every tree modification already takes O(tree_size) for the recursive
rebuild, this adds at most a constant factor per node during construction.

### Class API

```python
class RootedTree:
    __slots__ = (
        '_decoration', '_children', '_weight', '_tree_arity',
        '_leaves', '_min_leaf', '_arity', '_hash',
    )

    def __init__(self, decoration: tuple, *children):
        # children: tuple of (RootedTree | int)
        # Eagerly compute: weight, leaves, min_leaf, tree_arity
```

### Comparison & Hashing

- `__hash__`: Based on `(decoration, children)` tuple — structurally equivalent
  trees hash identically.
- `__eq__`: Recursive structural comparison (with short-circuit on hash mismatch).
- `__lt__`: Lexicographic comparison for SageMath basis ordering.

### Migration Strategy

1. Add `RootedTree` class to `trees.py`.
2. Update all functions in `trees.py` to accept and return `RootedTree`.
3. Add `RootedTree.from_tuple()` for backward compatibility.
4. Update bar/cobar constructions, tree_module, and morphisms.
5. Update tests.

### Operations on `RootedTree`

All existing operations are preserved with the same semantics:

- `relabel_leaves(tree, mapping)` → returns new `RootedTree`
- `graft(tree_top, i, tree_bot)` → returns new `RootedTree`
- `contract_edge(tree, parent, pos, dec)` → returns new `RootedTree`
- `expand_vertex(tree, target, pos, ...)` → returns new `RootedTree`
- `split_at_vertex(tree, target)` → returns `(RootedTree, int, RootedTree)`
- `vertices_dfs(tree)` → returns list of `RootedTree` references (identity-safe)

### Future Optimizations (Not in This PR)

- **Degree caching**: Store context-specific degrees (bar, cobar, free) via a
  `degree_cache` dict on each node. This would make `subtree_degree` O(1).
- **Persistent/structural sharing**: Use path-copying so that `contract_edge`
  only rebuilds nodes on the root-to-target path, sharing unchanged subtrees.
  This is already partially achieved since `RootedTree` children are references.
- **Flat array representation**: For very large trees, store nodes in a flat
  DFS-ordered array with parent/child indices for cache-friendly traversal.

---

## Status Update (2026-04): Planar Coinvariant Enumeration

In parallel with tree-level optimization, profiling showed a separate dominant
cost in cofree coalgebra basis enumeration: explicit `S_n` orbit-sum
construction for invariants. The current implementation now enumerates planar
representatives directly and canonicalizes boundary output back to planar keys.

### Mathematical Setup

For invariants of the form

`X ⊗_{S_n} Y` with `X = X_pl ⊗ k[S_n]`,

we use the canonical identification

`(X_pl ⊗ k[S_n]) ⊗_{S_n} Y  ≅  X_pl ⊗ Y`

to avoid explicit orbit sums in basis enumeration.

### Boundary Strategies Compared

Two methods were tested.

1. Orbit-sum method (reference):
   - Build orbit-sum representative.
   - Apply boundary.
   - Renormalize to planar representatives.
2. Planar-first method:
   - Keep only planar representative in the basis.
   - Apply boundary there.
   - Canonicalize non-planar output terms to planar keys.

### Result

The two methods are not coefficient-identical in general when the module tuple
`m` has repeated entries. The difference is a stabilizer factor `|Stab(m)|`.

- Over `QQ`, this appears as multiplicative factors (for example factor `2` in
  binary symmetry cases).
- Over positive characteristic, this factor can vanish (for example `2 = 0` in
  `GF(2)`), so matrix entries can differ by zero/nonzero behavior.

Despite this, the two complexes are chain-isomorphic under the coinvariant
identification above. The planar-first implementation satisfies `d^2 = 0` and
computes the same homology.

### Implementation Notes

- `basis_iter` / `basis_weight_iter` enumerate planar terms `term((c_pl, m))`
  directly (no orbit-sum construction during enumeration).
- Boundary matrix assembly canonicalizes non-planar terms via a full
  canonicalization routine that returns all planar contributions.
- This fixes a subtle lossy behavior from single-term canonicalization when
  planarization has multiple terms (for example with Lie-type factors).

### Validation Summary

- `d^2 = 0` checked at weights 2-4 over both `QQ` and `GF(2)`.
- Regression tests added for planar basis counting/canonical keys and boundary
  canonicalization behavior.
- Observed speedup on heavy layers around `2.8x` (example: 226s to 80s), with
  full test suite passing.
