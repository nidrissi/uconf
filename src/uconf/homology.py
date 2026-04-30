"""Chain-complex construction and homology helpers for dg-modules.

Given any module exposing ``graded_basis(d)`` and ``boundary`` (as used by
operad/cooperad components, bar/cobar constructions, free algebras, cofree
coalgebras, etc.), this module builds a SageMath
:class:`~sage.homology.chain_complex.ChainComplex` and provides helpers to
extract homology representatives as native module elements.

EXAMPLES::

    sage: from sage.all import QQ
    sage: from uconf import Surjection
    sage: from uconf.homology import compute_chain_complex, homology_basis
    sage: S2 = Surjection(2, QQ)
    sage: C = compute_chain_complex(S2, degrees=range(4))
    sage: 0 in C.betti()
    True
    sage: homology_basis(S2, degree=0)
    [{2 1}]
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any, Literal

from sage.all import ChainComplex, matrix

from uconf.core.signs import get_on_basis


_BOUNDARY_MATRIX_STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Boundary matrix
# ---------------------------------------------------------------------------


def _build_matrix_from_entries(
    base_ring: Any,
    n_target: int,
    n_source: int,
    entries: dict[tuple[int, int], Any],
    *,
    sparse: bool,
) -> Any:
    """Build a Sage matrix from sparse ``(row, col) -> coeff`` entries."""
    if not entries:
        return matrix(base_ring, n_target, n_source, sparse=sparse)
    if sparse:
        return matrix(base_ring, n_target, n_source, entries, sparse=True)
    M = matrix(base_ring, n_target, n_source, sparse=False)
    for (i, j), coeff in entries.items():
        M[i, j] = coeff
    return M


def _boundary_entries_for_key(
    key: Any,
    j: int,
    *,
    boundary_terms: Any,
    key_to_idx_target: dict,
    normalize_key: Any,
) -> dict[tuple[int, int], Any]:
    """Return the sparse matrix entries contributed by one source key."""
    entries: dict[tuple[int, int], Any] = {}
    for bdry_key, coeff in boundary_terms:
        i = key_to_idx_target.get(bdry_key)
        if i is not None:
            ij = (i, j)
            if ij in entries:
                entries[ij] += coeff
            else:
                entries[ij] = coeff
            continue
        if normalize_key is not None:
            found_match = False
            for norm_coeff, norm_key in normalize_key(bdry_key):
                i = key_to_idx_target.get(norm_key)
                if i is not None:
                    ij = (i, j)
                    combined = coeff * norm_coeff
                    if ij in entries:
                        entries[ij] += combined
                    else:
                        entries[ij] = combined
                    found_match = True
            if found_match:
                continue
        raise ValueError(
            f"Boundary of basis key {key!r} contains key {bdry_key!r} "
            "not found in target basis keys"
        )
    return entries


def _boundary_matrix_worker(task: tuple[int, int]) -> dict[tuple[int, int], Any]:
    """Compute sparse boundary entries for a chunk of source columns."""
    start, stop = task
    state = _BOUNDARY_MATRIX_STATE
    source_keys = state["source_keys"]
    boundary_on_basis = state["boundary_on_basis"]
    key_to_idx_target = state["key_to_idx_target"]
    normalize_key = state["normalize_key"]

    entries: dict[tuple[int, int], Any] = {}
    for j in range(start, stop):
        key = source_keys[j]
        column_entries = _boundary_entries_for_key(
            key,
            j,
            boundary_terms=boundary_on_basis(key),
            key_to_idx_target=key_to_idx_target,
            normalize_key=normalize_key,
        )
        for ij, coeff in column_entries.items():
            if ij in entries:
                entries[ij] += coeff
            else:
                entries[ij] = coeff
    return entries


def _chunk_ranges(n_items: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return contiguous chunk ranges covering ``range(n_items)``."""
    return [(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]


def _boundary_matrix(
    module: Any,
    basis_source_keys: list,
    key_to_idx_target: dict,
    n_target: int,
    *,
    sparse: bool,
    n_jobs: int = 1,
    basis_source: list | None = None,
) -> Any:
    """Build the matrix of the boundary map ``d: C_d -> C_{d-1}``.

    Parameters
    ----------
    module:
        The dg-module (must expose ``boundary``).
    basis_source:
        List of basis *elements* in the source degree.
    key_to_idx_target:
        Dictionary mapping basis *keys* in the target degree to row indices.
    n_target:
        Number of basis elements in the target degree (number of rows).

    Returns
    -------
    A SageMath matrix of size ``n_target × len(basis_source)`` over the
    base ring of *module*.
    """
    base_ring = module.base_ring()
    n_source = len(basis_source_keys)
    normalize_key = getattr(module, "_normalize_key", None)
    boundary_on_basis = get_on_basis(module.boundary)

    if boundary_on_basis is not None:
        if (
            n_jobs > 1
            and n_source > 0
            and os.name == "posix"
            and "fork" in multiprocessing.get_all_start_methods()
        ):
            chunk_size = max(1, (n_source + 8 * n_jobs - 1) // (8 * n_jobs))
            _BOUNDARY_MATRIX_STATE.clear()
            _BOUNDARY_MATRIX_STATE.update(
                {
                    "source_keys": basis_source_keys,
                    "boundary_on_basis": boundary_on_basis,
                    "key_to_idx_target": key_to_idx_target,
                    "normalize_key": normalize_key,
                }
            )
            entries: dict[tuple[int, int], Any] = {}
            ctx = multiprocessing.get_context("fork")
            try:
                with ctx.Pool(processes=n_jobs) as pool:
                    for chunk_entries in pool.imap_unordered(
                        _boundary_matrix_worker,
                        _chunk_ranges(n_source, chunk_size),
                    ):
                        for ij, coeff in chunk_entries.items():
                            if ij in entries:
                                entries[ij] += coeff
                            else:
                                entries[ij] = coeff
            finally:
                _BOUNDARY_MATRIX_STATE.clear()
            return _build_matrix_from_entries(
                base_ring,
                n_target,
                n_source,
                entries,
                sparse=sparse,
            )

        entries: dict[tuple[int, int], Any] = {}
        for j, key in enumerate(basis_source_keys):
            column_entries = _boundary_entries_for_key(
                key,
                j,
                boundary_terms=boundary_on_basis(key),
                key_to_idx_target=key_to_idx_target,
                normalize_key=normalize_key,
            )
            entries.update(column_entries)
        return _build_matrix_from_entries(base_ring, n_target, n_source, entries, sparse=sparse)

    if basis_source is None:
        raise ValueError(
            "basis_source elements are required when module.boundary lacks an on_basis fast path"
        )

    M = matrix(base_ring, n_target, n_source, sparse=sparse)
    for j, elem in enumerate(basis_source):
        for bdry_key, coeff in module.boundary(elem):
            i = key_to_idx_target.get(bdry_key)
            if i is not None:
                M[i, j] += coeff
                continue
            if normalize_key is not None:
                found_match = False
                for norm_coeff, norm_key in normalize_key(bdry_key):
                    i = key_to_idx_target.get(norm_key)
                    if i is not None:
                        M[i, j] += coeff * norm_coeff
                        found_match = True
                if found_match:
                    continue
            raise ValueError(
                f"Boundary of basis element {elem} contains key {bdry_key} "
                "not found in target basis keys"
            )
    return M


def _get_basis_elements(
    module: Any,
    d: int,
    weight: int | None = None,
    *,
    keep_elements: bool,
) -> tuple[list | None, list]:
    """Return ``(basis_elements_or_none, basis_keys)``.

    When *weight* is ``None``, we apply a ``connectivity`` short-circuit:
    if *d* is below the module's reported connectivity, the result is
    trivially empty.  When a *weight* filter is active we skip this check,
    because ``connectivity`` only captures the weight-1 minimum degree —
    higher-weight terms can reach lower degrees (e.g. when the cooperad in
    a cofree coalgebra has negative connectivity).
    """
    if weight is None:
        connectivity = getattr(module, "connectivity", None)
        if isinstance(connectivity, int) and d < connectivity:
            return [], []

    seen: set = set()
    elems: list | None = [] if keep_elements else None
    keys: list = []
    if weight is not None:
        family = module.graded_basis_by_weight(d, weight)
    else:
        family = module.graded_basis(d)
    for b in family:
        if hasattr(b, "leading_support"):
            key = b.leading_support()
        else:
            support = list(b.support())
            if not support:
                continue
            key = support[0]
        if key is None:
            continue
        if key not in seen:
            seen.add(key)
            if elems is not None:
                elems.append(b)
            keys.append(key)
        else:
            raise ValueError(
                f"Duplicate basis key {key} in degree {d} weight {weight} in {type(module).__name__}"
            )
    return elems, keys


def compute_chain_complex(
    module: Any,
    degrees: range,
    *,
    weight: int | None = None,
    check: bool = False,
    sparse: bool = True,
    n_jobs: int = 1,
) -> Any:
    """Build a SageMath :class:`ChainComplex` from a dg-module.

    Parameters
    ----------
    module:
        Any object exposing ``graded_basis(d)`` (returning a ``Family`` of
        basis elements), ``boundary`` (a linear map), and ``base_ring()``.
        All operad/cooperad components, bar/cobar constructions, free
        algebras, cofree coalgebras, and similar objects from *uconf*
        satisfy this interface.
    degrees:
        A :class:`range` of integer degrees.  The returned chain complex
        covers *degrees* plus one additional degree above (``max(degrees)+1``)
        so that the differential into ``max(degrees)`` is fully accounted for.
        Homology is correct for every degree in *degrees*; the Betti number
        at ``max(degrees)+1`` may be inflated by the truncation.
    weight:
        Optional fixed weight.  When provided, only basis elements of the
        given weight are included.  The module must expose
        ``graded_basis_by_weight(d, weight)``; if it does not, a
        :class:`ValueError` is raised.  Weight is the total number of
        "tensor factors" as defined by the module's ``_weight_on_basis``
        (for free algebras: the arity; for tree modules: the sum of leaf
        weights; for plain modules: the number of leaves).
    sparse:
        Whether to build differential matrices in sparse format.  This is
        usually faster and significantly lighter in memory for dg-modules
        with sparse boundaries.
    n_jobs:
        Number of worker processes to use for boundary-matrix assembly on the
        ``on_basis`` fast path.  Values above ``1`` currently use POSIX
        ``fork``-based multiprocessing and otherwise fall back to serial
        assembly.

    Returns
    -------
    A SageMath :class:`~sage.homology.chain_complex.ChainComplex` with
    ``degree_of_differential=-1``.

    EXAMPLES::

        sage: from sage.all import QQ
        sage: from uconf import Surjection
        sage: from uconf.homology import compute_chain_complex
        sage: S2 = Surjection(2, QQ)
        sage: C = compute_chain_complex(S2, degrees=range(4))
        sage: C.betti()[0]
        1
    """
    base_ring = module.base_ring()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}.")

    if not degrees:
        return ChainComplex({}, base_ring=base_ring, degree_of_differential=-1)

    if weight is not None and not hasattr(module, "graded_basis_by_weight"):
        raise ValueError(
            f"weight={weight} was specified but module {type(module).__name__} "
            "does not support the weight API.  "
            "Implement _weight_on_basis(key), basis_weight_iter(d, w), and "
            "graded_basis_by_weight(d, w) on the module first."
        )

    # Extend degrees by one in each direction to ensure we have the necessary basis elements to build the differentials for all degrees in the input range.  The chain complex will be correct in the input degrees; the extra degrees are just to ensure the differentials are correct and the homology computation can be performed without missing data.
    extended_degrees = range(min(degrees) - 1, max(degrees) + 2)

    # Collect basis elements and keys for each degree
    basis_by_degree: dict[int, list] = {}
    keys_by_degree: dict[int, list] = {}
    key_to_idx: dict[int, dict] = {}
    use_boundary_fast_path = get_on_basis(module.boundary) is not None

    for d in extended_degrees:
        basis_elems, basis_keys = _get_basis_elements(
            module,
            d,
            weight,
            keep_elements=not use_boundary_fast_path,
        )
        basis_by_degree[d] = basis_elems
        keys_by_degree[d] = basis_keys
        key_to_idx[d] = {k: i for i, k in enumerate(basis_keys)}

    # Build differential matrices: d_n : C_n -> C_{n-1}
    differentials: dict[int, Any] = {}
    for d in extended_degrees:
        if d - 1 not in key_to_idx:
            # Target degree not in range; if there are source basis elements,
            # we still need a zero matrix so the complex knows the rank of C_d.
            if basis_by_degree[d]:
                differentials[d] = matrix(base_ring, 0, len(basis_by_degree[d]), sparse=sparse)
            continue
        n_target = len(keys_by_degree[d - 1])
        source = basis_by_degree[d]
        if not source and n_target == 0:
            continue
        differentials[d] = _boundary_matrix(
            module,
            keys_by_degree[d],
            key_to_idx[d - 1],
            n_target,
            sparse=sparse,
            n_jobs=n_jobs,
            basis_source=source,
        )

    return ChainComplex(differentials, base_ring=base_ring, degree_of_differential=-1, check=check)


def compute_homology_representatives(
    module: Any,
    degree: int,
    weight: int | None,
    cc,
    *,
    algorithm: Literal["fast", "sage"] = "fast",
) -> list:
    """Compute cycle representatives for a basis of ``H_{degree}(cc)``.

    Two algorithms are available, controlled by the *algorithm* keyword:

    ``"fast"`` (default)
        Uses explicit linear algebra without invoking SageMath's Smith
        normal form machinery:

        1. Compute ``ker = ker(d_degree: C_degree → C_{degree-1})`` via
           :meth:`right_kernel` on the outgoing differential matrix.
        2. Compute ``im = im(d_{degree+1}: C_{degree+1} → C_degree)`` via
           the column space of the incoming differential.
        3. Row-reduce ``[im_basis; ker_basis]`` to echelon form.  The rows
           beyond the first ``dim(im)`` give homology representatives —
           they are in ``ker`` and their pivot columns are outside those of
           ``im``, so they are linearly independent modulo the image.

        This avoids the expensive ``generators=True`` path in SageMath and
        is significantly faster for large chain complexes.

    ``"sage"``
        Delegates to ``cc.homology(degree, generators=True)``.  Slower but
        produces representatives whose coefficients are expressed in
        SageMath's canonical reduced form, which can be easier to read.

    Args:
        module: The dg-module for which to compute representative cycles.
        degree: The homological degree at which to compute representatives.
        weight: Weight filter for basis selection, or ``None``.
        cc: A SageMath :class:`~sage.homology.chain_complex.ChainComplex`
            built from *module* via :func:`compute_chain_complex`.  Must
            contain differentials for *degree* and *degree*+1.
        algorithm: Which algorithm to use: ``"fast"`` (default) or
            ``"sage"``.

    Returns:
        A list of module elements that are cycles (``boundary(x) == 0``)
        whose homology classes form a basis of ``H_{degree}(cc)``.

    Raises:
        ValueError: If *algorithm* is not ``"fast"`` or ``"sage"``.
    """
    if algorithm == "sage":
        return _compute_homology_representatives_sage(module, degree, weight, cc)
    if algorithm == "fast":
        return _compute_homology_representatives_fast(module, degree, weight, cc)
    raise ValueError(f"Unknown algorithm {algorithm!r}: expected 'fast' or 'sage'.")


def _compute_homology_representatives_fast(
    module: Any, degree: int, weight: int | None, cc
) -> list:
    """Fast linear-algebra implementation; see :func:`compute_homology_representatives`."""
    basis_elems, _ = _get_basis_elements(module, degree, weight)
    if not basis_elems:
        return []

    # Step 1: ker(d_degree: C_degree -> C_{degree-1}).
    d_out = cc.differential(degree)
    ker = d_out.right_kernel()
    if ker.dimension() == 0:
        return []

    # Step 2: im(d_{degree+1}: C_{degree+1} -> C_degree).
    # cc.differential(degree+1) has shape dim(C_degree) × dim(C_{degree+1}).
    # Its column space is im(d_{degree+1}).
    d_in = cc.differential(degree + 1)
    K = ker.basis_matrix()  # k × n_degree, rows are cycles (row vectors)
    if d_in.ncols() == 0 or d_in.T.image().dimension() == 0:
        # im is trivial; homology = kernel
        rep_vecs = list(K.rows())
    else:
        I = d_in.column_space().basis_matrix()  # i × n_degree, rows span im(d_{d+1})
        # Step 3: row-reduce [I; K] so that the first dim(im) rows span im
        # and the remaining rows give independent homology representatives.
        E = I.stack(K).echelon_form()
        rank_I = I.rank()
        rep_vecs = [E.row(j) for j in range(rank_I, E.rank())]

    result: list = []
    for vec in rep_vecs:
        elem = module.zero()
        for i, coeff in enumerate(vec):
            if coeff:
                elem += coeff * basis_elems[i]
        result.append(elem)
    return result


def _compute_homology_representatives_sage(
    module: Any, degree: int, weight: int | None, cc
) -> list:
    """SageMath ``generators=True`` implementation; see :func:`compute_homology_representatives`."""
    basis_elems, _ = _get_basis_elements(module, degree, weight)

    ho = cc.homology(degree, generators=True)
    # cc.homology returns a bare tuple (not a list) when there are no
    # generators, so guard against that case.
    if not isinstance(ho, list):
        return []
    result: list = []
    for _vspace, gen in ho:
        vec = gen.vector(degree)
        elem = module.zero()
        for i, coeff in enumerate(vec):
            if coeff:
                elem += coeff * basis_elems[i]
        result.append(elem)
    return result


def homology_basis(
    module: Any,
    degree: int,
    *,
    degrees: range | None = None,
    weight: int | None = None,
) -> list:
    """Return cycle representatives for a basis of the homology in *degree*.

    Parameters
    ----------
    module:
        A dg-module (same requirements as :func:`compute_chain_complex`).
    degree:
        The homological degree in which to compute homology.
    degrees:
        Optional range of degrees to use when constructing the underlying
        chain complex.  Must include at least ``degree - 1``, ``degree``,
        and ``degree + 1`` so that both the incoming and outgoing
        differentials are available.  If ``None``, a minimal range
        ``range(degree - 1, degree + 2)`` is used (negative
        degrees are clamped to 0).  Pass a wider range if the module
        has non-trivial basis below ``degree - 1``.
    weight:
        Passed through to :func:`compute_chain_complex`. See its documentation.

    Returns
    -------
    A list of elements of *module* that are cycles (``boundary(x) == 0``)
    and whose homology classes form a basis of ``H_degree(module)``.

    EXAMPLES::

        sage: from sage.all import QQ
        sage: from uconf import Surjection
        sage: from uconf.homology import homology_basis
        sage: S2 = Surjection(2, QQ)
        sage: homology_basis(S2, 0)
        [{2 1}]
    """
    if degrees is None:
        degrees = range(degree - 1, degree + 2)
    else:
        if degree not in degrees:
            raise ValueError(f"degree {degree} must be contained in the supplied range {degrees}")

    C = compute_chain_complex(module, degrees, weight=weight)

    return compute_homology_representatives(module, degree, weight, C)
