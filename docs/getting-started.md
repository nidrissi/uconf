# Getting started

## Build requirements

The documentation build imports `uconf`, so it must run in the same SageMath
environment used for tests and linting.

Install the package with the documentation dependency group (the `--group` flag
requires pip ≥ 25.1):

```bash
conda run -n sage python -m pip install --upgrade pip
conda run -n sage python -m pip install -e . --group docs
```

Build the HTML site:

```bash
conda run -n sage sphinx-build --keep-going -b html docs docs/_build/html
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
