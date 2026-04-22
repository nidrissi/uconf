# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`uconf` is a SageMath-based Python library for combinatorial operad/cooperad models used in algebraic topology computations (configuration spaces, bar/cobar constructions, E-comodule structures). The accompanying math article is `article.tex`/`article.bib` — do not edit those files.

## Setup

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -e ".[dev]"
```

The `--system-site-packages` flag is required to inherit SageMath from the system.

## Commands

```bash
# Run all tests (parallel via pytest-xdist)
pytest

# Run a single test file
pytest tests/test_surjection.py

# Run a single test function
pytest tests/test_surjection.py::test_compose

# Lint and format
ruff check src/
ruff format src/

# Benchmark
python benchmark.py
```

Tests run with `-q -n auto` by default (quiet, all cores). Always activate `.venv` first.

## Architecture

The package lives entirely in `src/uconf/`. Canonical imports are subpackage-based (e.g. `from uconf.models.surjection import Surjection`), though `uconf/__init__.py` re-exports everything publicly.

**Layer structure:**

- `core/` — protocols and shared utilities: `OperadComponent`/`CooperadComponent` protocols, `OperadMorphism`, `TwistingMorphism`, `trees.py` (immutable `RootedTree` with eagerly cached structural properties), `signs.py`, `quasi_planar.py`.
- `models/` — concrete operad/cooperad implementations: `Surjection`, `BarrattEccles`, `Lie`, `Associative`, `Commutative`, `SurjectionDual`, `CoAssociative`, `CoCommutative`, `SimplicialChains`, `SimplicialCochains`.
- `wrappers/` — operad/cooperad wrappers: `ShiftedOperad`, `ShiftedCooperad`, `HadamardProduct`.
- `constructions/` — bar/cobar functors: `BarConstruction`, `CobarConstruction`, `BarAlgebra`, `CobarCoalgebra`.
- `algebraic/` — algebra/coalgebra wrappers: `FreeOperadAlgebra`, `CofreeConilpotentCoalgebra`, `HadamardTensorAlgebra`, `PullbackAlgebra`, simplicial/spherical models, configuration models.
- `morphisms/` — concrete morphisms: `ass_to_com`, `lie_to_ass`, `canonical_projection`, `canonical_inclusion`, `make_e_comodule_morphism`.
- `homology.py` — `compute_chain_complex`, `homology_basis`, `compute_homology_representatives`.
- `sampling.py` — construction-aware random samplers; used extensively in tests to avoid full basis enumeration.

**Dynamic wiring at `import uconf`:** `BarrattEccles.Element.table_reduction` and `Surjection.Element.section` are attached as lazy module morphisms in `uconf/__init__.py`. Do not replicate this wiring elsewhere.

## Key invariants

- **All operads and cooperads are connected**: P(0) = 0, P(1) = k·unit. This is a hard requirement, not a special case — it ensures finite (arity, degree) bases in bar/cobar constructions.
- **Component classes are arity-fixed** (`self._arity`). Each component is a separate SageMath `CombinatorialFreeModule`.
- **Constructors normalize degenerate keys to 0; malformed keys raise.** `dict` input gives linear combinations; tuple/list input gives a basis element.
- **Planar coinvariant model:** For quasi-planar factors, free/cofree modules enumerate planar representatives instead of full S_n orbit sums, using the isomorphism `(P_pl ⊗ k[S_n]) ⊗_{S_n} M^⊗n ≅ P_pl ⊗ M^⊗n`. The boundary canonicalizes non-planar output back to planar keys.
- **Permutations in tests use one-line list notation**, e.g. `[2, 1]` for the transposition in S_2.

## Coding conventions

- Use **SageMath-native types** (`Integer`, `Rational`, ring elements), not Python `int`/`float`. Methods like `term` and `sum_of_terms` do not coerce coefficients — wrong types silently produce wrong results.
- Internal boundary computations use `_from_validated_tree` to skip validation for known-valid inputs. Use `module.term(key)` rather than `module(key)` for already-validated keys.
- Accumulate terms via `{key: coeff}` dicts + single `sum_of_terms()` call rather than repeated `result += coeff * module.term(key)`.
- Hot-path methods (`_boundary_on_basis`, `_d1_on_basis`, `_d2_on_basis`, `_dalpha_on_basis`) are cached with `@cached_method`.

## When making changes

- Edits go in `src/uconf/`; tests go in `tests/`.
- If a public API or usage pattern changes, update `README.md` in the same commit.
- Run `ruff check` and relevant tests before finalizing.
- Mathematical correctness takes priority: if a test fails due to a mathematical inconsistency, fix the math rather than the test.
- See `README.md` for full API documentation including usage examples, bar/cobar tree conventions, and test coverage map.
- See `docs/optimizations.md` for performance optimization history and current bottlenecks.
