# Najib and Victor Project

Combinatorial operad/cooperad models (SageMath) for computations in algebraic topology and configuration spaces.

## Repository structure

- `src/uconf/`: main implementation.
  - `core/`: protocols and shared utilities (`operad`, `cooperad`, `signs`, `trees`).
  - `models/`: concrete operad/cooperad/simplicial models.
  - `algebraic/`: algebra/coalgebra wrappers, free/cofree constructions.
  - `constructions/`: bar/cobar constructions and algebraic bar/cobar complexes.
  - `wrappers/`: shifted and Hadamard operad/cooperad wrappers.
- `tests/`: regression test suite.
- `docs/`: Sphinx documentation sources.
- `benchmark.py`, `homology_repr.py`: CLI scripts for configuration-model computations (write artifacts to `dump/`).
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

