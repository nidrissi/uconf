# Najib and Victor Project

Combinatorial operad/cooperad models (SageMath) for computations in algebraic topology and configuration spaces.

## Repository structure

- `src/uconf/`: main implementation (active API).
  - `src/uconf/core/`: protocols and shared utilities (`operad`, `cooperad`, `signs`, `trees`).
  - `src/uconf/models/`: concrete operad/cooperad/simplicial models.
  - `src/uconf/algebraic/`: algebra/coalgebra wrappers, free/cofree constructions.
  - `src/uconf/constructions/`: bar/cobar constructions and algebraic bar/cobar complexes.
  - `src/uconf/wrappers/`: shifted and Hadamard operad/cooperad wrappers.
  - `src/uconf/tikz.py`: TikZ/forest rendering of bar/cobar tree elements.
  - `src/uconf/tex/uconf-trees.sty`: LaTeX style file with `forest` macros for tree pictures.
- `tests/test_*.py`: main regression test suite.
- `pyproject.toml`: packaging plus pytest/ruff configuration.
- `docs/`: Sphinx documentation sources.
- `benchmark.py`, `homology_repr.py`: CLI scripts for configuration-model computations.
- `article.tex`, `article.bib`: project-related scientific writing.
- `old-computations/`: older notebooks/utilities kept for reference.

## Prerequisites

**SageMath** is required. `pytest` is needed to run the tests. `comch` is optional (used only by `test_comch_compatibility.py`).

## Development

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
conda run -n sage pytest
```

Validate before committing:

```bash
conda run -n sage ruff check tests src
conda run -n sage ruff format --check tests src
conda run -n sage python -m compileall -q src tests
```

## HTML documentation

Install docs dependencies and build:

```bash
conda run -n sage python -m pip install -e ".[docs]"
conda run -n sage sphinx-build --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html/`.

## `uconf` package

Canonical imports are subpackage-based (e.g., `uconf.models.surjection`).

**All (co)operads are connected:** P(0) = 0 and P(1) = 𝑘·unit.  This ensures
every bar/cobar basis is finite without an external weight cap.  Each class
exposes a `connectivity: int` attribute (constant 𝑘 such that P(𝑛) lives in
degrees ≥ 𝑘·(𝑛−1)).

### Operad/cooperad models

| Module | Type | Notes |
|---|---|---|
| `models/surjection.py` | `Surjection` | Non-degenerate surjective words. Simplicial action via `uconf.algebraic.simplicial`. |
| `models/barratt_eccles.py` | `BarrattEccles` | Sequences of non-consecutive permutations. |
| `models/lie.py` | `Lie` | Hall-basis model with PBW caches. |
| `models/surjection_dual.py` | `SurjectionDual` | Linear dual of `Surjection`. |
| `models/simplicial.py` | `SimplicialChains`, `SimplicialCochains` | Normalized chains/cochains on standard simplices. |
| `wrappers/shifted_operad.py` | `ShiftedOperad(P, d)` | Arity-dependent degree shift with compatible signs. |
| `wrappers/shifted_cooperad.py` | `ShiftedCooperad(C, d)` | Cooperadic shift wrapper. |
| `wrappers/hadamard_operad.py` | `HadamardProduct(P, Q)` | Aritywise tensor product `(P⊙Q)(n) = P(n)⊗Q(n)`. |

### Bar/cobar constructions

| Module | Type | Notes |
|---|---|---|
| `constructions/bar_construction.py` | `BarConstruction(P)` | `B(P) = (T^c(s\bar{P}), d_1+d_2)`. Quasi-planar when P is (e.g. `Surjection`, `BarrattEccles`). |
| `constructions/cobar_construction.py` | `CobarConstruction(C)` | `Ω(C) = (T(s⁻¹\bar{C}), d_1+d_2)`. |
| `constructions/bar_algebra.py` | `BarAlgebra(alpha, alg)` | `B_α(A) = T^c_C(A)` for a twisting morphism `α: C→P` and a P-algebra `A`. |
| `constructions/cobar_coalgebra.py` | `CobarCoalgebra(alpha, coalg)` | `Ω_α(V) = T_P(V)` for a C-coalgebra `V`. |

Constructors taking operads/cooperads accept either a class (e.g. `Lie`) or a
wrapper instance (e.g. `ShiftedOperad(Lie, -1)`, `HadamardProduct(Associative, Associative)`).

### Morphisms

| Module | Exports |
|---|---|
| `morphisms/classical.py` | `ass_to_com`, `lie_to_ass` |
| `morphisms/canonical_twisting.py` | `canonical_projection(P)`, `canonical_inclusion(C)` |
| `morphisms/e_comodule_morphism.py` | `make_e_comodule_morphism(cooperad_cls)` — builds `Δ: Ω(C) → E⊗Ω(C)` |
| `algebraic/pullback_algebra.py` | `PullbackAlgebra(morphism, algebra)` |

### Chain complexes and homology (`homology.py`)

- `compute_chain_complex(module, degrees, *, weight=None, sparse=True, n_jobs=1, progress=False, ...)` — builds a SageMath `ChainComplex` from any dg-module that exposes `graded_basis(d)`, `boundary`, and `base_ring()`.  Use `weight` to restrict free/cofree modules to a finite subcomplex.  `n_jobs>1` parallelizes matrix assembly via POSIX `fork`.

- `homology_basis(module, degree, *, degrees=None, weight=None)` — returns cycle representatives whose classes form a basis of `H_degree(module)`.

- `compute_homology_representatives(module, degree, weight, cc, *, algorithm="fast")` — given a pre-built chain complex, returns explicit cycle representatives.

```python
from sage.all import QQ
from uconf import Surjection
from uconf.homology import compute_chain_complex, homology_basis

S2 = Surjection(2, QQ)
C = compute_chain_complex(S2, degrees=range(5))
C.homology()
# {0: Vector space of dimension 1 over Rational Field, ...}

homology_basis(S2, 0, degrees=range(5))
# [S2[(2, 1)]]
```

For the configuration model, use `weight` to compute a finite subcomplex:

```python
from sage.all import QQ
from uconf import euclidean_unordered_configuration_model
from uconf.homology import compute_chain_complex

model = euclidean_unordered_configuration_model(QQ, 2)
C = compute_chain_complex(model.module, degrees=range(-1, 4), weight=1, n_jobs=8)
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

## TikZ / forest rendering of tree elements

`src/uconf/tikz.py` exposes `element_to_tikz` (re-exported from `uconf`) which converts a bar / cobar / cofree-coalgebra element into a compact LaTeX snippet using the `forest` package.  The companion style file `src/uconf/tex/uconf-trees.sty` provides the styles and a `uconf tree` preset; copy or symlink it somewhere on your LaTeX search path and load it once in your document:

```latex
\usepackage{uconf-trees}
```

Default layer styling (selected automatically by nesting depth):

| layer                       | vertex style | edge style       |
| --------------------------- | ------------ | ---------------- |
| outer bar (red)             | `rv`         | `re` (red)       |
| cobar                       | `bv`         | `de` (dashed)    |
| inner bar inside cobar      | `bv`         | default (solid)  |
| manifold / coefficient cell | `bx`         | default          |
| leaf generator              | `lf`         | default          |

Example:

```python
from sage.all import QQ
from uconf import BarConstruction, Lie, element_to_tikz
from uconf.core.trees import RootedTree

B2 = BarConstruction(Lie)(2, QQ)
elem = B2(RootedTree((1,), 1, 2))
print(element_to_tikz(elem))
# \begin{forest} uconf tree
# [{...}, rv[{1}, lf, re][{2}, lf, re]]
# \end{forest}
```

The `homology_repr.py` script accepts `--tikz` to dump a `.tex` file with one `forest` block per representative, ready to be `\input`'d into a document that loads `uconf-trees`.

For finer control, `tree_to_forest(tree, decoration_formatter=..., layer=...)` renders a single decorated `RootedTree` with explicit styling, and `Layer` lets you mix and match vertex/edge styles.

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

## Top-level scripts

Two CLI scripts drive end-to-end computations on the Euclidean unordered
configuration model.  Both write artifacts to `dump/` with filenames of the
form `F<p>_d<dim>_w<weight>_m<deg_max>_*` (or `Q_...` for rationals).

### `benchmark.py` — assemble the chain complex

Builds the configuration-model chain complex and saves: chain complex
(`*_cc.sobj`), Betti numbers (`*_cc.csv`), graded bases (`*_bases.sobj`), and
(unless `--no-profile`) a `cProfile` report.

```bash
python benchmark.py --dim 2 --weight 2 --deg_max 3 --jobs 8
python benchmark.py --field 3 --dim 2 --weight 2 --deg_max 3
python benchmark.py --field Q --dim 2 --weight 2 --deg_max 3
```

Key arguments:

- `--dim, -d` (default `2`) — sphere dimension.
- `--weight, -w` (default `2`) — weight of the configuration subcomplex.
- `--deg_max, -m` (default `3`) — maximum degree.
- `--field, -f` (default `2`) — base field: prime power `p` for `GF(p)`, or `Q`/`QQ`.
- `--jobs, -j` (default `1`) — parallel workers for matrix assembly.
- `--verbose, -v` — timestamped phase diagnostics to stderr.
- `--no-prewarm` — disable cache prewarm before forking.
- `--no-profile` — skip `cProfile`.

### `homology_repr.py` — extract homology representatives

Loads a chain complex dump from `benchmark.py` and computes explicit cycle
representatives via `compute_homology_representatives`.  Saves a text report
(`*_homology_reps.txt`) and a pickle of monomial-coefficient dicts
(`*_homology_reps.sobj`).

```bash
python homology_repr.py dump/F2_d2_w2_m3_cc.sobj
python homology_repr.py dump/Q_d2_w2_m3_cc.sobj --algorithm fast
```

The script infers parameters from the dump filename; pass `--yes`/`-y` to
skip the confirmation prompt.

Key arguments:

- `dump` (positional) — path to the `.sobj` chain complex file.
- `--dim`, `--weight`, `--deg_max`, `--field` — override inferred parameters.
- `--deg_min` (default `-1`) — minimum degree.
- `--algorithm, -a` (`fast` or `sage`, default `fast`).
- `--yes, -y` — skip confirmation.
- `--verbose, -v`, `--no-profile` — as in `benchmark.py`.

The pickle stores `{degree: [monomial_coefficients_dict, ...]}` (not module
elements directly, because `BarAlgebraModule` elements contain closures that
cannot be pickled).  To reconstruct an element from its dict `mc`:

```python
e = sum((coeff * mod.monomial(key) for key, coeff in mc.items()), mod.zero())
```

