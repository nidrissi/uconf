# Getting started

## Build requirements

The documentation build imports `uconf`, so it must run in the same SageMath
environment used for tests and linting.

Install the package with the documentation extra:

```bash
conda run -n sage python -m pip install -e ".[docs]"
```

Build the HTML site:

```bash
conda run -n sage sphinx-build --keep-going -b html docs docs/_build/html
```

The generated site is written to
`/home/runner/work/najib-victor/najib-victor/docs/_build/html`.

## What is included

- narrative overview pages for the main package areas,
- generated API reference pages for the `uconf` package and subpackages,
- the existing project notes already stored in `docs/`.

## What is not wired yet

- publishing to GitHub Pages or Read the Docs,
- doctest execution for the Sage-flavoured examples in docstrings,
- custom theming beyond the default Sphinx theme.
