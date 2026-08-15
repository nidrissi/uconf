# Getting started

## Build requirements

The documentation build imports `uconf`, so it must run in the same SageMath
environment used for tests and linting.

### Conda environment

Install the package with the documentation dependency group (the `--group` flag
requires pip ≥ 25.1):

```bash
conda run -n sage python -m pip install --upgrade pip
conda run -n sage python -m pip install -e . --group docs
```

Build the HTML site with conda:

```bash
conda run -n sage sphinx-build --keep-going -b html docs docs/_build/html
```

### uv with an existing SageMath installation

For local development, uv treats SageMath as an externally managed dependency.
The published package still declares `sagemath>=10.9`, but the
`exclude-dependencies` setting in `pyproject.toml` prevents `uv sync` from
downloading SageMath and its transitive dependencies. The virtual environment
must instead inherit SageMath from the interpreter on which it is built:

```bash
uv venv --system-site-packages
uv sync --group docs
uv run python -c 'from importlib.metadata import version; print(version("sagemath"))'
```

Use `uv venv --python <path> --system-site-packages` when SageMath belongs to a
non-default interpreter. Since uv excludes SageMath from resolution, the last
command must report version 10.9 or later. Then build with:

```bash
uv run sphinx-build --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`.

## What is included

- narrative overview pages for the main package areas,
- generated API reference pages for the `uconf` package and subpackages,
- the existing project notes already stored in `docs/`.

## What is not wired yet

- publishing to GitHub Pages or Read the Docs,
- doctest execution for the Sage-flavoured examples in docstrings,
- custom theming beyond the default Sphinx theme.
