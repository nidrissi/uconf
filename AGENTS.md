# AI Developer Guide for `uconf`

`uconf` is a SageMath-based Python library for combinatorial operad/cooperad
models in algebraic topology. This file tells AI agents how to make changes
safely. Read it in full before your first edit.

## 1. Halt protocol — read this first

Some failures look like bugs but are mathematical decisions in disguise. A
wrong sign or coefficient is a one-character edit that compiles, passes ruff,
and is silently wrong. **Halt and ask the user** — write a `TODO` in code and
stop — whenever a test failure or required edit touches any of:

- Signs, suspension conventions, or anything in `core/signs.py`.
- Boundary or differential definitions (`_boundary_on_basis`,
  `_d1_on_basis`, bar/cobar `d_1 + d_2`).
- Twisting morphisms, Maurer–Cartan terms (`core/twisting.py`).
- Coinvariant / planar canonicalization logic (see §5).
- Connectedness assumptions in bar/cobar constructions (see §5).
- The dynamic wiring in `uconf/__init__.py` (see §5).

Programmatic fixes — typos, imports, type errors, missing arguments,
ruff-flagged style — handle on your own. Anything mathematical: halt.

## 2. Environment

Two paths exist; they are not stale relative to each other, they are for
different agent contexts. Pick the one that matches your runtime.

### GitHub Copilot cloud agent → conda

SageMath lives in a conda environment named `sage`. Prefix every Python or
tool command with `conda run -n sage`:

```bash
conda run -n sage pytest
conda run -n sage ruff check tests src
conda run -n sage ruff format --check tests src
conda run -n sage python -m compileall -q src tests
```

### Local development → venv

SageMath is installed system-wide; the venv inherits it via
`--system-site-packages`. One-time setup:

```bash
python3 -m venv .venv --system-site-packages   # flag goes on venv, not pip
source .venv/bin/activate
pip install -e ".[dev]"
```

After activation, run `pytest`, `ruff check tests src`,
`ruff format --check tests src`, and `python -m compileall -q src tests`
directly.

Targeted test run (either environment, prefix as needed):

```bash
pytest tests/test_file.py::test_function
```

## 3. Read-only files and directories

Do not edit:

- `article.tex`, `article.bib` — publication source, not code. Do not edit unless the user explicitly asks you to.
- `old-computations/` — kept for historical reference; do not refactor.
- `dump/` — output directory for `benchmark.py` and `homology_repr.py`.
  Agents may write artifacts here when running the CLI scripts, but do not
  curate or restructure its contents, and do not rename existing dumps —
  `homology_repr.py` parses filenames to infer parameters (see §7).

## 4. Required validation before finalizing

Run all four. They mirror CI. Substitute the environment prefix from §2.

```bash
<env> ruff check tests src
<env> ruff format --check tests src
<env> python -m compileall -q src tests
```

Ruff is configured to lint **both** `tests/` and `src/`. Linting only `src/`
is incomplete and will fail CI.

Afterwards, run the validation on the tests that are affected by the changes you made. Do not run the full test suite (which is time-expensive) unless the user specifically asks you to.

## 5. Mathematical invariants

These are non-negotiable architectural assumptions. Code relies on them; do
not introduce special cases that violate them.

### Connectedness

Every operad and cooperad in this package is connected: $P(0) = 0$ and
$P(1) = k \cdot \text{unit}$, and dually $C(0) = 0$ and
$C(1) = k \cdot \text{counit}$. This bounds the arity-$n$ component of any
bar/cobar construction by $n-1$ internal vertices and makes every
`(arity, degree)` basis finite — which is why no external `max_weight` cap
is needed. Breaking this breaks termination, not just correctness.

Each operad/cooperad class exposes a `connectivity: int` attribute (the
constant $k$ such that $P(n)$ is concentrated in degrees $\geq k(n-1)$).
Respect it when composing wrappers; `ShiftedOperad(P, d).connectivity` is
`P.connectivity + d`, and similarly for `HadamardProduct`.

### Planar coinvariant model (free / cofree)

Free and cofree modules enumerate **planar representatives**, not full
$S_n$-orbit sums, using the identification
$(X_{\text{pl}} \otimes k[S_n]) \otimes_{S_n} Y \cong X_{\text{pl}} \otimes Y$.

Boundaries can produce non-planar terms; these **must** be canonicalized
back to planar keys, with coefficients redistributed across all planar
contributions when the planarization is multi-term. The implemented complex
is chain-isomorphic to the orbit-sum complex and satisfies $d^2 = 0$, but
coefficients differ from the orbit-sum version by $|\text{Stab}(m)|$ when
the inner-module tuple has repeated entries — which matters in positive
characteristic (e.g., $2 = 0$ in $\mathrm{GF}(2)$). Do not "simplify" by
removing the canonicalization step.

### Dynamic wiring at `import uconf`

Two maps are attached lazily in `uconf/__init__.py` via `module_morphism`:

- `BarrattEccles.Element.table_reduction() -> Surjection.Element`
- `Surjection.Element.section() -> BarrattEccles.Element`

Leave both where they are. Do not move them to class bodies, do not eagerly
construct them at module import, and do not add new attachments alongside
without coordinating with the user.

### Provider API for operads and cooperads

Constructors that take an operad or cooperad accept **either** the class
(e.g., `Lie`) **or** a wrapper instance (e.g., `ShiftedOperad(Lie, -1)`,
`HadamardProduct(Associative, Associative)`, `BarConstruction(Lie)`). This
is the `OperadLike` / `CooperadLike` contract in `core/operad.py` and
`core/cooperad.py`. New constructors should accept the same — do not
hard-code "must be a class" or "must be an instance."

## 6. Coding conventions

**SageMath types only.** Use `Integer`, `Rational`, or specific base-ring
elements — never bare Python `int` or `float` for mathematical scalars. In
particular, `CombinatorialFreeModule.sum_of_terms` does not coerce Python
scalars, and silent coefficient errors are easy to introduce this way.
(Convention; if you're touching coercion-sensitive code, verify the
specific method's behavior in the source before relying on edge cases.)

**Term accumulation pattern.** Build a `{key: coeff}` dict and call
`sum_of_terms(...)` once at the end. Avoid repeated
`result += coeff * module.term(key)` loops — they re-hash and re-validate
on every iteration.

**Cache hot paths.** Methods like `_boundary_on_basis` and `_d1_on_basis`
are decorated with `@cached_method`. Preserve the decorator when
refactoring. (Convention; check the existing decorator on the method
you're editing — not every internal method is cached.)

**Internal fast paths.** Constructors that bypass validation
(e.g., `_from_validated_tree`) exist for known-valid inputs inside boundary
code. Use them only on values whose validity you can prove locally; the
validated public path everywhere else.

**Permutation notation in tests.** Use one-line list notation: `[2, 1]`,
not cycle notation.

**Tree encoding.** Trees are nested tuples. Leaves are integers in
$\{1, \ldots, n\}$. Internal vertices are
`(decoration, child_1, ..., child_k)`. Internal arity is $\geq 2$ by the
connectedness invariant.

## 7. Surfaces worth knowing before you touch them

The patterns here are not obvious from a single file.

### Homology / chain complex API (`uconf.homology`)

`compute_chain_complex(module, degrees, *, weight=None, sparse=True, n_jobs=1, …)`:

- `weight` is **required** for modules with infinite bases (free algebras,
  cofree coalgebras, configuration-model bar modules). Without it, basis
  enumeration does not terminate.
- `sparse=True` (default) is substantially faster on typical boundaries.
- `n_jobs > 1` uses POSIX `fork`-based multiprocessing. Anything held in
  module-import-time state is inherited by workers; cache prewarm
  (`prewarm=True`, default) materializes hot caches before forking. Do not
  silently disable it.

### CLI scripts

`benchmark.py` and `homology_repr.py` at the repo root drive
configuration-model computations end-to-end. Artifacts go to `dump/` with a
filename scheme `F<p>_d<dim>_w<weight>_m<deg_max>_*` (e.g.
`F2_d2_w2_m3_cc.sobj`, `Q_d2_w2_m3_homology_reps.txt`). `homology_repr.py`
parses this back out to infer parameters from a dump path. **Do not rename
the scheme** without updating the inference logic in `homology_repr.py`.

### Pickle constraint

`BarAlgebraModule` elements contain closures and cannot be pickled. The
convention is to persist `{key: coeff}` monomial-coefficient dicts and
reconstruct elements on load:

```python
e = sum((coeff * mod.monomial(key) for key, coeff in mc.items()), mod.zero())
```

Preserve this convention if you add new persistence or serialization code.
Do not try to pickle module elements directly — it will appear to work in
small cases and break on closures.
