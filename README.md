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

