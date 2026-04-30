# Najib and Victor Project

Combinatorial operad/cooperad models (SageMath) for computations in algebraic topology and configuration spaces.

## Repository structure

- `src/uconf/`: main implementation (active API).
  - `src/uconf/core/`: protocols and shared utilities (`operad`, `cooperad`, `signs`, `trees`).
  - `src/uconf/models/`: concrete operad/cooperad/simplicial models.
  - `src/uconf/algebraic/`: algebra/coalgebra wrappers, free/cofree constructions.
  - `src/uconf/constructions/`: bar/cobar constructions and algebraic bar/cobar complexes.
  - `src/uconf/wrappers/`: shifted and Hadamard operad/cooperad wrappers.
- `tests/test_*.py`: main regression test suite.
- `pyproject.toml`: packaging plus pytest/ruff configuration.
- `docs/`: project notes and optimization writeups.
- `old-computations/`: older notebooks/utilities kept for reference.
- `article.tex`, `article.bib`: project-related scientific writing.

## Prerequisites

The project relies on **SageMath** (parents/modules, symmetric groups, tensor products, etc.).

- Key dependency: `sagemath`.
- Tests: `pytest`.
- Optional: `comch` for compatibility tests (`test_comch_compatibility.py`).

## Development

Create and activate a virtual environment, then install the package in development mode:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"
```

Then run tests:

```bash
conda run -n sage pytest
```

Useful validation commands:

```bash
conda run -n sage ruff check tests src
conda run -n sage ruff format --check tests src
conda run -n sage python -m compileall -q src tests
```

## HTML documentation

The repository now includes a Sphinx configuration in
`/home/runner/work/najib-victor/najib-victor/docs` that builds HTML
documentation from the existing docstrings.

Install the documentation dependencies:

```bash
conda run -n sage python -m pip install -e ".[docs]"
```

Build the site:

```bash
conda run -n sage sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated HTML files are written to `docs/_build/html/`.

## `uconf` package

Canonical imports are subpackage-based (e.g., `uconf.models.surjection`).

### Connected (co)operads — global assumption

**All operads and cooperads in this package are connected:** P(0) = 0 and
P(1) = 𝑘·unit (resp. counit).  This is a hard requirement rather than a
special case.

Connectedness ensures that every internal vertex of a bar/cobar tree has
arity ≥ 2, bounding the number of vertices in an arity-𝑛 component by 𝑛 − 1.
This makes every (arity, degree) basis finite and removes the need for an
external `max_weight` cap in bar/cobar constructions.

Each operad/cooperad class exposes a `connectivity: int` attribute (class
attribute on concrete models, property on wrappers) representing the constant
𝑘 such that P(𝑛) is concentrated in degrees ≥ 𝑘·(𝑛−1):

| Class | `connectivity` |
|---|---|
| `Surjection`, `BarrattEccles`, `Lie`, `Commutative`, `Associative`, … | 0 |
| `ShiftedOperad(P, d)` | `P.connectivity + d` |
| `ShiftedCooperad(C, d)` | `C.connectivity + d` |
| `HadamardProduct(P, Q)` | `P.connectivity + Q.connectivity` |

### Operad models

- `models/surjection.py` — `Surjection`
  - Basis: non-degenerate surjective words (no consecutive repetitions).
  - Constructor semantics: tuples with consecutive repetitions or missing labels map to zero; malformed labels/types raise.
  - Operations: `unit`, `compose`, `boundary`, `permute`, `complexity`, `planar_basis_iter`.
  - Acts on simplicial models through wrappers in `uconf.algebraic.simplicial`.

- `models/barratt_eccles.py` — `BarrattEccles`
  - Basis: sequences of permutations in `S_n` with no consecutive duplicates.
  - Constructor semantics: tuples with consecutive duplicate permutations map to zero; malformed permutation data raises.
  - Operations: `unit`, `compose`, `boundary`, `permute`, `diagonal`, `planarize`.

- `models/lie.py` — `Lie`
  - Hall-basis model (nested brackets).
  - Operations: `unit`, `compose`, `permute` (antisymmetry/Jacobi behavior).
  - Includes PBW change-of-basis caches to accelerate `compose`.

### Cooperad model

- `models/surjection_dual.py` — `SurjectionDual`
  - Linear dual companion of `Surjection`.
  - Operations: `counit`, `reduced`, `infinitesimal_cocompose`.

### Bar-cobar constructions

- `constructions/bar_construction.py` — `BarConstruction(P)`
  - Bar construction of a connected dg-operad: `B(P) = (T^c(s\bar{P}), d_1 + d_2)`.
  - Cooperadic model on decorated rooted trees.
  - Differential combines internal differential on vertex decorations and edge-contraction terms.
  - **Quasi-planar structure** (when the base operad is quasi-planar, e.g. `Surjection`, `BarrattEccles`, or a `HadamardProduct` whose right factor is quasi-planar):
    - `Component.planarize(x)` — decomposes `x` into `x_pl ⊗ σ ∈ B(P)_pl(n) ⊗ k[S_n]`.
    - `Component.planar_basis_iter(d)` — iterates over strongly-planar basis trees of bar degree `d`.
    - `Component.basis_iter(d)` — iterates over **all** shuffle-tree basis elements of bar degree `d`.
    - `Component.d_sigma(x, σ)` — the `σ`-component of `boundary(x)`, i.e. `π_σ(d(x))`.
    - `Component.d_sigma_iterate(x, [σ₁,…,σₖ])` — iterated `d_σ` with zero-branch pruning.
    - `Element.planarize()` — convenience wrapper.

- `constructions/cobar_construction.py` — `CobarConstruction(C)`
  - Cobar construction of a connected dg-cooperad: `\Omega(C) = (T(s^{-1}\bar{C}), d_1 + d_2)`.
  - Operadic model on decorated rooted trees.
  - Differential combines internal differential and vertex-expansion terms from infinitesimal cocomposition.
  - `Component.basis_iter(d)` — iterates over shuffle-tree basis elements of cobar degree `d`.
    Works for any connected cooperad, including those with negative-degree elements (e.g.
    `CoAssociative`).

- `constructions/bar_algebra.py` — `BarAlgebra(alpha, alg)`
  - Given a twisting morphism `α: C → P` and a P-algebra `A`, constructs the
    bar construction `B_α(A) = T^c_C(A)` with differential `d = d_cofree + d_α`.
  - Basis keys are pairs `(c_key, (a_1, …, a_n))` where `c_key ∈ C(n)` is a cooperad
    basis key and `a_i` are algebra module basis keys.
  - Wraps `CofreeCoalgebraModule` and adds the `d_α` coderivation.
  - Exposes C-coalgebra coaction via `bar.coact(elem, n)`.
  - Standard bar complex: `BarAlgebra(canonical_projection(P), alg)`
  - Twisted bar complex B_ι(A): `BarAlgebra(canonical_inclusion(B(P)), alg)`

- `constructions/cobar_coalgebra.py` — `CobarCoalgebra(alpha, coalg)`
  - Given a twisting morphism `α: C → P` and a C-coalgebra `V`, constructs the
    cobar construction `Ω_α(V) = T_P(V)` with differential `d = d_free + d_α`.
  - Basis keys are pairs `(p_key, (v_1, …, v_n))` where `p_key ∈ P(n)` is an operad
    basis key and `v_i` are coalgebra module basis keys.
  - Wraps `FreeAlgebraModule` and adds the `d_α` derivation.
  - Exposes P-algebra action via `cobar.act(p_elem, elems)`.
  - Standard cobar complex: `CobarCoalgebra(canonical_inclusion(C), coalg)`

- `constructions/comodule.py` — `e_comodule_on_generator`
  - Implements the `E_ν`-comodule structure `Δ: Ω(C) → E_ν ⊗ Ω(C)` on planar generators of
    the cobar construction of a quasi-planar cooperad `C`.
  - Formula on a planar generator `s⁻¹x ∈ s⁻¹C_pl(n)`:

    ```
    Δ(s⁻¹x) = Σ_{k≥0} Σ_{σ̄∈(Sₙ\{id})^k} ρ(σ̄) ⊗ cobar(d_{σ̄}(x))·σ₁…σₖ
    ```

    where `ρ(σ̄) = (id, σₖ, σₖ₋₁σₖ, …, σ₁…σₖ) ∈ E(n)` and `d_{σ̄} = d_{σ₁}∘…∘d_{σₖ}`.
  - The sum terminates at `k = deg(x)` (degree truncation).
  - Zero branches pruned early for efficiency.

### Operad morphisms and pullbacks

- `core/morphism.py` — `OperadMorphism`
  - `OperadMorphism(source, target, on_element)`: wraps a linear map between operad
    components into a morphism `f: P → Q`.

- `morphisms/classical.py` — `ass_to_com`, `lie_to_ass`
  - `ass_to_com`: augmentation morphism `Ass → Com` (sends every permutation `σ ∈ S_n`
    to the commutative generator).
  - `lie_to_ass`: PBW inclusion `Lie → Ass` (sends Lie brackets to commutator expansions).

- `morphisms/e_comodule_morphism.py` — `make_e_comodule_morphism(cooperad_cls)`
  - Builds the operad morphism `Δ: Ω(C) → E ⊗ Ω(C)` for a quasi-planar cooperad `C`.
  - On generators (weight-1 cobar trees), delegates to `e_comodule_on_generator`.
  - On arbitrary trees, extends via the universal property of the free operad:
    `Δ(T) = Δ(gen_root) ∘_k Δ(child_k) ∘ … ∘_1 Δ(child_1)`.
  - Target operad is `HadamardProduct(BarrattEccles, CobarConstruction(C))`.

- `core/pullback_algebra.py` — `PullbackAlgebra'
  - `PullbackAlgebra(morphism, algebra)`: given a `Q`-algebra and a morphism `f: P → Q`,
    produces a `P`-algebra whose structure map is `γ^P(p; a_1,…,a_n) = γ^Q(f(p); a_1,…,a_n)`.

### Operadic twisting morphisms

- `core/twisting.py` — `TwistingMorphism(cooperad, operad, morphism_fn)`
  - A degree -1 map `α: C̄ → P` satisfying the Maurer-Cartan equation `∂α + α ⋆ α = 0`.
  - `alpha(c_elem)`: apply the morphism to a cooperad element.
  - `alpha.star(beta, c_elem)`: compute the pre-Lie convolution product `(α ⋆ β)(c)`.
  - `alpha.partial_alpha(c_elem)`: compute `∂α(c) = ∂_P(α(c)) + α(∂_C(c))`.
  - `alpha.maurer_cartan(c_elem)`: evaluate `∂α + α ⋆ α` on a single element.
  - `alpha.check_maurer_cartan(max_arity, base_ring)`: verify MC via d² = 0 on the twisted
    bar complex.

- `morphisms/canonical_twisting.py`
  - `canonical_projection(P)`: the canonical projection `π: B(P) → P`, which sends a bar
    corolla to its P-decoration and zero on multi-vertex trees.
  - `canonical_inclusion(C)`: the canonical inclusion `ι: C → Ω(C)`, which sends `c ∈ C(n)`
    to the single-vertex cobar tree `(c, 1, …, n)`.
  - These induce the standard adjunctions:
    - `π`: B_π ⊣ Ω_π between P-algebras and B(P)-coalgebras
    - `ι`: B_ι ⊣ Ω_ι between Ω(C)-algebras and C-coalgebras

- `core/trees.py`
  - Shared rooted-tree combinatorics used by bar/cobar modules.
  - Utilities for DFS traversal, arity/weight/leaves, grafting, edge contraction, and vertex expansion.
  - `enumerate_shuffle_trees_in_degree(arity, weight_bound, P, R, d)` — bar degree `Σ(deg_P+1)`.
  - `enumerate_shuffle_trees_free_in_degree(arity, weight_bound, P, R, d)` — free degree `Σ deg_P`.
    Used by `FreeAlgebraModule.basis_iter` and `CofreeCoalgebraModule.basis_iter`.
  - `enumerate_shuffle_trees_cobar_in_degree(arity, weight_bound, C, R, d)` — cobar degree `Σ(deg_C-1)`.
    Used by `CobarConstruction.Component.basis_iter`.

### Chain complexes and homology (`homology.py`)

- `compute_chain_complex(module, degrees, *, weight=None, check=False, sparse=True)` — builds a SageMath
  `ChainComplex` from any dg-module that exposes `graded_basis(d)`,
  `boundary`, and `base_ring()`.  This includes all operad/cooperad
  components, bar/cobar constructions, free algebras, cofree coalgebras,
  and similar objects.

  Use `sparse=True` (the default) for substantially faster and lighter
  differential matrix assembly on typical sparse boundaries.

  For free algebras, cofree coalgebras, and configuration-model bar modules,
  the `weight` parameter restricts to a fixed finite-weight summand, giving a
  tractable subcomplex when unrestricted basis enumeration would be infinite.

- `homology_basis(module, degree, *, degrees=None, weight=None)` — returns
  a list of elements of *module* that are cycles and whose homology classes
  form a basis of `H_degree(module)`.  When *degrees* is not given, a
  minimal range `[degree-1, degree+1]` is used.

```python
from sage.all import QQ
from uconf import Surjection
from uconf.homology import compute_chain_complex, homology_basis

S2 = Surjection(2, QQ)
C = compute_chain_complex(S2, degrees=range(5))
C.homology()
# {0: Vector space of dimension 1 over Rational Field,
#  1: …dimension 0…, 2: …dimension 0…, 3: …dimension 0…,
#  4: Vector space of dimension 1 over Rational Field}

homology_basis(S2, 0, degrees=range(5))
# [S2[(2, 1)]]
```

For the configuration model, use `weight` to compute a finite subcomplex:

```python
from sage.all import QQ
from uconf import euclidean_unordered_configuration_model
from uconf.homology import compute_chain_complex

model = euclidean_unordered_configuration_model(QQ, 2)
# `model` is a BarAlgebra; compute_chain_complex works on its underlying dg-module.
C = compute_chain_complex(model.module, degrees=range(-1, 4), weight=1)
C.betti()
# {-1: 0, 0: 0, 1: 0, 2: 1, 3: 0}
```

### Simplicial models

- `models/simplicial.py`
  - `SimplicialChains`: normalized chains on standard simplices, basis = non-degenerate
    simplex tuples `(v_0, …, v_n)` (strictly-increasing non-negative integers).
    - Constructor semantics: empty simplices and simplices with consecutive repeated vertices map to zero; malformed simplex data raises.
    - `SimplicialChains.fundamental_chain(n, base_ring)` — the fundamental cycle `[0,…,n]`.
    - `SimplicialChains.basis_iter(base_ring, N)` — iterator over all simplices in `Δ^N`.
    - `Element.boundary()` — simplicial boundary on arity-1 elements.
    - `Element.iterated_diagonal(times)` — AW diagonal; returns a native Sage
      `tensor([SimplicialChains(base_ring)]*(times+1))` element.
  - `SimplicialCochains(N, base_ring)`: dual cochains on `Δ^N`, same simplex-tuple basis as
    `SimplicialChains`.
    - Constructor semantics: empty simplices and simplices with consecutive repeated vertices map to zero; malformed simplex data and vertices outside `\{0, ..., N\}` raise.
    - `SimplicialCochains.volume_form(N, base_ring)` — the volume form on `Δ^N`.
    - `SimplicialCochains.evaluate(cochain, chain)` — Kronecker pairing.
    - `SimplicialCochains.dual_basis_it(N, base_ring)` — iterator over dual basis.
    - `Element.coboundary()` — coboundary operator.

### Surjection action/coaction API

- Canonical implementation lives in `uconf.algebraic.simplicial`:
  - `surjection_chain_action(u, x)` — action of `u ∈ S(r)` on a chain `x ∈ C`; returns
    a native `tensor([SimplicialChains(base_ring)]*r)` element (`r ≥ 2`) or `SimplicialChains`
    element (`r = 1`).
  - `surjection_cochain_action(u, (f_1, …, f_r))` — dual cochain action.
  - `SurjectionSimplicialChainCoalgebra(base_ring=QQ)` — coalgebra wrapper
    on simplicial chains
  - `SurjectionSimplicialCochainAlgebra(N, base_ring=QQ)` — algebra wrapper
    on simplicial cochains

### Surjection action examples

```python
from sage.all import QQ, tensor
from uconf import SimplicialChains, SimplicialCochains, Surjection
from uconf.algebraic.simplicial import (
    SurjectionSimplicialChainCoalgebra,
    SurjectionSimplicialCochainAlgebra,
    surjection_chain_action,
)

# Chains
SC = SimplicialChains(QQ)
x = SC((0, 1, 2))            # the 2-simplex [0,1,2]
x.boundary()                  # ∂[0,1,2]
x.iterated_diagonal(times=1) # Δ([0,1,2]) ∈ tensor([SC, SC])

# Tensor-product boundary (for tensor([SC]*r) elements)
T = tensor([SC, SC])
y = x.iterated_diagonal(times=1)
surjection_chain_action(Surjection(2, QQ)((1, 2, 1)), x)

# Coalgebra (chain-side) wrapper
coalg = SurjectionSimplicialChainCoalgebra(base_ring=QQ)
u = Surjection(2, QQ)((1, 2, 1))  # degree-1 surjection
coalg.coact(x, 2)             # δ_2(x) ∈ SurjectionDual(2) ⊗ SC ⊗ SC

# Cochains
Cco = SimplicialCochains(N=3, base_ring=QQ)
f = Cco((0, 1))               # the dual cochain [0,1]*
SimplicialCochains.evaluate(f, SC((0, 1)))  # 1

# Algebra (cochain-side) wrapper
alg = SurjectionSimplicialCochainAlgebra(N=3, base_ring=QQ)
alg.act(u, [f, f])            # μ_u(f⊗f) ∈ SimplicialCochains(N=3)
```

### Protocols and utilities

- `core/operad.py` — `OperadComponent` (minimal structural contract for operads).
- `core/cooperad.py` — `CooperadComponent` (dual cooperadic contract).
- `core/operad.py` — `OperadLike` accepts either an operad class (for example
  `Lie`) or an operad factory/wrapper instance (for example
  `ShiftedOperad(Lie, -1)` and `HadamardProduct(Associative, Associative)`).
- `core/cooperad.py` — `CooperadLike` similarly accepts cooperad classes or
  cooperad wrapper instances (for example `BarConstruction(Lie)`).
- `core/signs.py` — shared sign conventions (permutation signature, suspension signs).

### Wrappers

- `wrappers/shifted_operad.py` — `ShiftedOperad(P, d)`
  - Arity-dependent degree shift.
  - Sign twists for `boundary`, symmetric action, and `compose`.

- `wrappers/shifted_cooperad.py` — `ShiftedCooperad(C, d)`
  - Cooperadic shift wrapper with compatible sign rules.

- `wrappers/hadamard_operad.py` — `HadamardProduct(P, Q)`
  - Aritywise Hadamard product: `(P ⊙ Q)(n) = P(n) ⊗ Q(n)`.
  - Tensor differential: `d(a⊗b)=da⊗b+(-1)^|a|a⊗db`.
  - Diagonal symmetric action and diagonal composition.
  - `Component.basis_iter(d)` — iterates over all `(left_key, right_key)` pairs with total degree `d`.
  - `Component.planar_basis_iter(d)` — if the right factor `Q` has `planar_basis_iter`,
    iterates over pairs `(left_key, right_pl_key)` with right key planar and total degree `d`.

- `algebraic/hadamard_algebra.py` — `HadamardTensorAlgebra(A, B)`
  - Input: a `P`-algebra `A` and a `Q`-algebra `B`.
  - Output: a `(P ⊙ Q)`-algebra on `tensor([A.module, B.module])`.
  - Action is diagonal on factors and multilinear on tensor arguments.
  - `basis_iter(d)` — iterates over all `(left_key, right_key)` tensor basis elements with total degree `d`.

- `algebraic/free_algebra.py` — `FreeOperadAlgebra(operad_cls, inner_module)`
  - Free P-algebra on a dg-module M: `P ∘ M = ⊕_{n≥1} P(n) ⊗_{S_n} M^{⊗n}`.
  - Basis keys are `(tree, m_tuple)` pairs where `tree` is a shuffle tree and
    `m_tuple` is a tuple of M-basis keys.
  - Degree: `deg(tree, m_tuple) = Σ_v deg_P(dec(v)) + Σ_i deg_M(m_i)` (no suspension).
  - `FreeAlgebraModule.basis_iter(d)` — iterates over all basis elements of degree `d`.
    The arity is bounded automatically from M's minimum degree and P's connectivity.
    Works correctly when M has elements only in strictly-positive degrees or P has
    `connectivity ≥ 1`; otherwise it raises `ValueError` rather than returning a
    partial list.
  - For quasi-planar inputs, `basis_iter` uses planar representatives rather than
    explicit `S_n`-orbit sums. This keeps the same coinvariant information while
    avoiding expensive orbit construction.

- `algebraic/cofree_coalgebra.py` — `CofreeConilpotentCoalgebra(cooperad_cls, inner_module)`
  - Cofree conilpotent C-coalgebra on a dg-module M: `T^c_C(M) = ⊕_{n≥1} C(n) ⊗_{S_n} M^{⊗n}`.
  - Same basis key convention as `FreeAlgebraModule` (shuffle trees + M-tuple).
  - `CofreeCoalgebraModule.basis_iter(d)` — iterates over planar representatives
    `(c_pl, m)` of basis elements of degree `d` (instead of precomputing full orbit
    sums);
    same arity-bounding logic as `FreeAlgebraModule.basis_iter` and the same
    `ValueError` fail-fast behavior in non-exhaustive regimes.
  - The boundary matrix canonicalizes non-planar output terms back to planar keys,
    redistributing coefficients across all planar contributions when planarization
    is multi-term.
- `algebraic/spherical.py`
  - `ReducedSphereCochains(d)` — rank-1 module for reduced cochains of `S^d`,
    concentrated in degree `d`.
  - `SurjectionSphereCochainAlgebra(d)` — explicit `Surjection`-algebra structure
    on `ReducedSphereCochains(d)`, following the sign/concatenation criterion
    from Proposition~\ref{prop:surj-alg-sphere} in `article.tex`.

## Bar-cobar tree conventions

- Trees are encoded as nested tuples.
  - Leaves are integers in `{1, ..., n}`.
  - Internal vertices are `(decoration, child_1, ..., child_k)`.
- In the current implementation, constructions assume connected inputs (`\bar{P}(1)=0`, `\bar{C}(1)=0`), so internal vertex arity is at least `2`.
- Degree bookkeeping follows suspension/desuspension conventions used in `bar_construction.py` and `cobar_construction.py`.

## Dynamic wiring at `import uconf`

In `uconf/__init__.py`, two maps are attached dynamically (lazy + parent-level cache):

- `BarrattEccles.Element.table_reduction() -> Surjection.Element`
- `Surjection.Element.section() -> BarrattEccles.Element`

These maps are built via `module_morphism` on first use.

## Important conventions

- Component classes have a **fixed arity** (`self._arity`).
- Constructors generally accept `dict` (linear combinations) and tuple/list (basis key).
- Degenerate keys are normalized to `0` through internal validation; malformed keys raise `TypeError` or `ValueError`.
- For permutations in tests, use **one-line list notation** (for example `[2, 1]`).

## Planar coinvariant model (free/cofree)

When a factor splits as `X = X_pl ⊗ k[S_n]`, free/cofree modules use the
identification

`(X_pl ⊗ k[S_n]) ⊗_{S_n} Y  ≅  X_pl ⊗ Y`

to enumerate only planar representatives. This removes the dominant orbit-sum
cost in basis enumeration.

Two boundary strategies were analyzed:

1. Orbit-sum first, then boundary, then renormalize (coefficient-level canonical).
2. Boundary on planar representatives, then canonicalize non-planar outputs.

These are not coefficient-identical in general when `m_tuple` has repeated
entries: coefficients differ by `|Stab(m)|`. In particular, over positive
characteristic this factor can vanish (for example, `2 = 0` in `GF(2)`).

The implemented complex uses planar representatives plus boundary
canonicalization. It is chain-isomorphic to the orbit-sum complex, satisfies
`d^2 = 0`, and has the same homology, while being substantially faster in
practice.

## Quick examples

### Shifted operad

```python
from uconf import Lie, ShiftedOperad

ShiftLie = ShiftedOperad(Lie, 1)
L2 = ShiftLie(2)
x = L2((1,))
y = x.permute([2, 1])
z = ShiftLie.compose(x, 2, x)
```

### Hadamard product

```python
from uconf import HadamardProduct, Lie, Surjection

Had = HadamardProduct(Lie, Surjection)
H2 = Had(2)

x = H2(((1,), (1, 2)))
y = H2(((1,), (1, 2, 1)))

z = Had.compose(x, 1, y)
dx = x.boundary()
xp = x.permute([2, 1])
```

### Hadamard tensor algebra

```python
from uconf import HadamardTensorAlgebra

AB = HadamardTensorAlgebra(A, B)  # A: P-algebra, B: Q-algebra
u = AB.operad_cls.unit()          # unit in (P ⊙ Q)(1)
t = AB.module.term((a_key, b_key))

AB.act(u, [t])
```

### Bar / cobar constructions

```python
from uconf import BarConstruction, CobarConstruction, Lie, SurjectionDual

# Bar construction B(Lie)
BLie = BarConstruction(Lie)
B3 = BLie(3)
t = B3(((1, 2), 1, 2, 3))
dt = t.boundary()

# Cobar construction Ω(S*)
OmegaS = CobarConstruction(SurjectionDual)
O2 = OmegaS(2)
x = O2(((1, 2), 1, 2))
u = OmegaS.unit()

# Free-operad composition (tree grafting)
comp = OmegaS.compose(x, 1, u)

# Cooperadic infinitesimal cocomposition on bar elements
delta = t.infinitesimal_cocompose(i=2, m=2, n=2)
```

Constructors taking operads/cooperads now use the same provider API, so nested
wrappers are accepted directly:

```python
from uconf import BarConstruction, CobarConstruction, HadamardProduct, Lie, ShiftedOperad, Surjection

s_lie = ShiftedOperad(Lie, -1)
surj_s_lie = HadamardProduct(Surjection, s_lie)
omega_b = CobarConstruction(BarConstruction(surj_s_lie))
```

### Surjection action on simplicial chains/cochains

```python
from sage.all import QQ
from uconf import SimplicialChains, Surjection
from uconf.algebraic.simplicial import (
    SurjectionSimplicialChainCoalgebra,
    surjection_chain_action,
)

u = Surjection(2, QQ)((1, 2, 1))
x = SimplicialChains.fundamental_chain(3, QQ)
theta = surjection_chain_action(u, x)
coalg = SurjectionSimplicialChainCoalgebra(QQ)
delta = coalg.coact(x, 2)
```

```python
from sage.all import QQ
from uconf import SimplicialCochains, Surjection
from uconf.algebraic.simplicial import SurjectionSimplicialCochainAlgebra

u = Surjection(2, QQ)((1, 2, 1))
C = SimplicialCochains(N=3, base_ring=QQ)
f1 = C((0, 1))
f2 = C((1, 2))
alg = SurjectionSimplicialCochainAlgebra(N=3, base_ring=QQ)
mu = alg.act(u, [f1, f2])
```

### Operad morphisms and pullback algebras

```python
from uconf import Associative, Commutative, Lie, OperadMorphism, PullbackAlgebra
from uconf import ass_to_com, lie_to_ass
from sage.all import QQ

# Apply the Ass → Com augmentation morphism
x = Associative(3, QQ)((2, 1, 3))
ass_to_com(x)  # → Com(3) generator

# Apply the Lie → Ass PBW inclusion
bracket = Lie(2, QQ)((1,))
lie_to_ass(bracket)  # → (1,2) - (2,1) in Ass(2)

# Compose morphisms: Lie → Ass → Com kills all brackets
ass_to_com(lie_to_ass(bracket))  # → 0

# Pullback a Com-algebra along Ass → Com to get an Ass-algebra
# com_alg = OperadAlgebra(module, Commutative, structure_map)
# ass_alg = PullbackAlgebra(ass_to_com, com_alg)
# ass_alg.act(mu, [a, a])  # delegates to com_alg.act(ass_to_com(mu), [a, a])
```

### E-comodule morphism Δ: Ω(C) → E ⊗ Ω(C)

```python
from uconf import (
    BarConstruction, CobarConstruction, HadamardProduct,
    BarrattEccles, Lie, make_e_comodule_morphism,
)
from sage.all import QQ

# Build the cooperad B(Lie ⊙ E) and its cobar construction
HLE = HadamardProduct(Lie, BarrattEccles)
BH = BarConstruction(HLE)
OBH = CobarConstruction(BH)

# Build the morphism Δ: Ω(B(Lie⊙E)) → E ⊗ Ω(B(Lie⊙E))
Delta = make_e_comodule_morphism(BH)

# Apply to a cobar element
unit = OBH.unit(QQ)
Delta(unit)  # → unit of HadamardProduct(BE, Ω(B(Lie⊙E)))
```

## Tests (coverage)

### API contracts

- `test_common_operad.py`: `OperadComponent` conformance (`Surjection`, `BarrattEccles`).
- `test_common_cooperad.py`: `CooperadComponent` conformance (`SurjectionDual`).

### Main operads

- `test_surjection.py`: units, symmetric action, composition, planar bases, section/table-reduction.
- `test_barratt_eccles.py`: basis cardinalities, unit, symmetric action, composition.
- `test_lie.py`: unit, antisymmetry, Jacobi, operadic axioms, stress checks in arities 4–6.

### Cooperad and wrappers

- `test_surjection_cooperad.py`: `counit`, `reduced`, duality with `compose` via `infinitesimal_cocompose`.
- `test_shifted_operad.py`: sign/degree twists and README smoke test.
- `test_shifted_cooperad.py`: cooperadic shift behavior (`counit`, cocomposition signs).
- `test_hadamard_operad.py`: additive degree, tensor-differential sign rule, diagonal action/composition.
- `test_bar_cobar.py`: tree utilities, bar/cobar differentials (`d_1 + d_2`), basic composition/cocomposition behavior, and `\partial^2 = 0` checks on sample elements.

### Morphisms and pullbacks

- `test_morphisms.py`:
  - `ass_to_com`: unit preservation, equivariance, composition compatibility, chain-map property, linearity.
  - `lie_to_ass`: unit preservation, bracket-to-commutator, equivariance, composition compatibility.
  - `PullbackAlgebra`: pullback of Com-algebra along `Ass → Com`, unit axiom, boundary delegation.
  - `make_e_comodule_morphism`: unit preservation, output type, generator agreement with `e_comodule_on_generator`.

### Chain complexes and homology

- `test_homology.py`:
  - `compute_chain_complex`: Surjection/BarrattEccles/Lie Betti numbers, `d²=0` check, empty degrees.
  - `homology_basis`: cycle verification, dimension checks, default/invalid degree ranges.

### Simplicial and external compatibility

- `test_simplicial.py`:
  - chain/cochain validity,
  - `∂²=0`,
  - AW diagonal chain-map property,
  - surjection action/coaction,
  - chain-cochain adjointness (pairing checks).
- `test_comch_compatibility.py` (if `comch` is installed): operation-by-operation comparison with `comch`.
- `test_stress_operads.py`: deterministic randomized tests (linearity, unit, `∂²=0`, Jacobi).

## Useful commands

- Run all tests:
  - `conda run -n sage pytest`
- Run wrapper-focused tests:
  - `conda run -n sage pytest -q tests/test_shifted_operad.py tests/test_shifted_cooperad.py tests/test_hadamard_operad.py`
- Run bar/cobar tests:
  - `conda run -n sage pytest -q tests/test_bar_cobar.py`
- Run core operad tests:
  - `conda run -n sage pytest -q tests/test_protocols.py tests/test_surjection.py tests/test_barratt_eccles.py tests/test_lie.py`
- Run simplicial tests:
  - `conda run -n sage pytest -q tests/test_simplicial.py`
- Run morphism tests:
  - `conda run -n sage pytest -q tests/test_morphisms.py`
