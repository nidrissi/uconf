# Getting started

## Build requirements

The documentation build imports `uconf`, so it must run in the same SageMath
environment used for tests and linting.

### SageMath Docker image

This is what CI does. `uv venv --system-site-packages` cannot be used inside the
image — Sage's interpreter is itself a virtualenv, so a child venv resolves its
base to the system Python and misses Sage entirely. Install the locked
documentation dependencies into Sage's interpreter instead:

```bash
docker run --rm -it -v "$PWD":/work -w /work sagemath/sagemath:10.9 bash
sage -pip install uv   # the image ships no uv, curl or wget
SAGEPY=$(sage -python -c 'import sys; print(sys.executable)')
sage -python -m uv export --frozen --no-hashes --no-emit-project --group docs -o /tmp/reqs.txt
sage -python -m uv pip install --python "$SAGEPY" -r /tmp/reqs.txt
sage -python -m uv pip install --python "$SAGEPY" --no-deps -e .
```

Build the HTML site:

```bash
sage -python -m sphinx --keep-going -b html docs docs/_build/html
```

### uv with an existing SageMath installation

For local development, uv treats SageMath as an externally managed dependency.
The published package still declares `sagemath>=10.7`, but the
`exclude-dependencies` setting in `pyproject.toml` prevents `uv sync` from
downloading SageMath and its transitive dependencies. The virtual environment
must instead inherit SageMath from the interpreter on which it is built:

```bash
uv venv --system-site-packages
uv sync --group docs
uv run python -c 'import sage.version; print(sage.version.version)'
```

Use `uv venv --python <path> --system-site-packages` when SageMath belongs to a
non-default interpreter. Since uv excludes SageMath from resolution, the last
command must report version 10.7 or later. Then build with:

```bash
uv run sphinx-build --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html`.

## What is included

- narrative overview pages for the main package areas,
- generated API reference pages for the `uconf` package and subpackages,
- the existing project notes already stored in `docs/`,
- native Sage doctest execution for the examples in package docstrings.

## What is not wired yet

- publishing to GitHub Pages or Read the Docs,
- custom theming beyond the default Sphinx theme.
