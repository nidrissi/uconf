# Najib and Victor Project

Combinatorial operad/cooperad models (SageMath) for computations in algebraic topology and configuration spaces.

## Repository structure

- `calculs/uconf/`: main implementation (active API).
- `calculs/test_*.py`: main regression test suite.
- `calculs/pytest.ini`: pytest configuration (`-q`, `test_*.py`).
- `calculs/misc.py`, `calculs/misc.ipynb`: drafts/experiments.
- `old-computations/`: older notebooks/utilities kept for reference.
- `article.tex`, `article.bib`: project-related scientific writing.

## Prerequisites

The project relies on **SageMath** (parents/modules, symmetric groups, tensor products, etc.).

- Key dependency: `sagemath` (see `requirements.txt`).
- Tests: `pytest`.
- Optional: `comch` for compatibility tests (`test_comch_compatibility.py`).

## `calculs/uconf` package

### Operad models

- `surjection.py` — `Surjection`
  - Basis: non-degenerate surjective words (no consecutive repetitions).
  - Operations: `unit`, `compose`, `boundary`, `permute`, `complexity`, `planar_basis_it`.
  - Extra actions: `act` on `SimplicialChains`, `coact` on `SimplicialCochains`.

- `barratt_eccles.py` — `BarrattEccles`
  - Basis: sequences of permutations in `S_n` with no consecutive duplicates.
  - Operations: `unit`, `compose`, `boundary`, `permute`, `diagonal`, `planarize`.

- `lie.py` — `Lie`
  - Hall-basis model (nested brackets).
  - Operations: `unit`, `compose`, `permute` (antisymmetry/Jacobi behavior).
  - Includes PBW change-of-basis caches to accelerate `compose`.

### Cooperad model

- `surjection_linear_dual.py` — `SurjectionLinearDual`
  - Linear dual companion of `Surjection`.
  - Operations: `counit`, `reduced`, `infinitesimal_cocompose`.

### Simplicial models

- `simplicial.py`
  - `SimplicialChains`: normalized chains on standard simplices, tensor differential, iterated AW diagonal.
  - `SimplicialCochains`: dual cochains, coboundary, Kronecker pairing (`evaluate`).

### Protocols and utilities

- `operad.py` — `OperadProtocol` (minimal structural contract for operads).
- `cooperad.py` — `CooperadProtocol` (dual cooperadic contract).
- `signs.py` — shared sign conventions (permutation signature, suspension signs).

### Wrappers

- `shifted_operad.py` — `ShiftedOperad(P, d)`
  - Arity-dependent degree shift.
  - Sign twists for `boundary`, symmetric action, and `compose`.

- `shifted_cooperad.py` — `ShiftedCooperad(C, d)`
  - Cooperadic shift wrapper with compatible sign rules.

- `hadamard_operad.py` — `HadamardOperad(P, Q)`
  - Aritywise Hadamard product: `(P ⊙ Q)(n) = P(n) ⊗ Q(n)`.
  - Tensor differential: `d(a⊗b)=da⊗b+(-1)^|a|a⊗db`.
  - Diagonal symmetric action and diagonal composition.

## Dynamic wiring at `import uconf`

In `uconf/__init__.py`, two maps are attached dynamically (lazy + parent-level cache):

- `BarrattEccles.Element.table_reduction() -> Surjection.Element`
- `Surjection.Element.section() -> BarrattEccles.Element`

These maps are built via `module_morphism` on first use.

## Important conventions

- Component classes have a **fixed arity** (`self._arity`).
- Constructors generally accept `dict` (linear combinations) and tuple/list (basis key).
- Degenerate/invalid keys are normalized to `0` through internal validation.
- For permutations in tests, use **one-line list notation** (for example `[2, 1]`).
- Run pytest from `calculs/` so `import uconf` resolves naturally.

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
from uconf import HadamardOperad, Lie, Surjection

Had = HadamardOperad(Lie, Surjection)
H2 = Had(2)

x = H2(((1,), (1, 2)))
y = H2(((1,), (1, 2, 1)))

z = Had.compose(x, 1, y)
dx = x.boundary()
xp = x.permute([2, 1])
```

### Surjection action on simplicial chains

```python
from uconf import SimplicialChains, Surjection

u = Surjection(2)((1, 2, 1))
x = SimplicialChains.standard_element(3)
res = Surjection.act(u, x)
```

## Tests (coverage)

### API contracts

- `test_common_operad.py`: `OperadProtocol` conformance (`Surjection`, `BarrattEccles`).
- `test_common_cooperad.py`: `CooperadProtocol` conformance (`SurjectionLinearDual`).

### Main operads

- `test_surjection.py`: units, symmetric action, composition, planar bases, section/table-reduction.
  - Includes known `xfail` tests for subtle `section` compatibilities.
- `test_barratt_eccles.py`: basis cardinalities, unit, symmetric action, composition.
- `test_lie.py`: unit, antisymmetry, Jacobi, operadic axioms, stress checks in arities 4–6.

### Cooperad and wrappers

- `test_surjection_cooperad.py`: `counit`, `reduced`, duality with `compose` via `infinitesimal_cocompose`.
- `test_shifted_operad.py`: sign/degree twists and README smoke test.
- `test_shifted_cooperad.py`: cooperadic shift behavior (`counit`, cocomposition signs).
- `test_hadamard_operad.py`: additive degree, tensor-differential sign rule, diagonal action/composition.

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

From `calculs/`:

- Run all tests:
  - `pytest`
- Run wrapper-focused tests:
  - `pytest -q test_shifted_operad.py test_shifted_cooperad.py test_hadamard_operad.py`
- Run core operad tests:
  - `pytest -q test_common_operad.py test_surjection.py test_barratt_eccles.py test_lie.py`
- Run simplicial tests:
  - `pytest -q test_simplicial.py`

## Notes

- `calculs/uconf/` is the current source of truth.
- `old-computations/` is historical material, not the primary target of the current test suite.
