"""Chain complex construction and homology helpers for dg-modules.

Given any module exposing ``graded_basis(d)`` and ``boundary`` (as used by
operad/cooperad components, bar/cobar constructions, free algebras, cofree
coalgebras, etc.), this module builds a SageMath :class:`~sage.homology.chain_complex.ChainComplex`
and provides helpers to extract homology representatives as native module
elements.

Typical usage::

    >>> from uconf import Surjection
    >>> from uconf.homology import chain_complex, homology_basis
    >>> S2 = Surjection(2, QQ)
    >>> C = chain_complex(S2, degrees=range(4))
    >>> C.homology()
    >>> homology_basis(S2, degree=0)
"""

from __future__ import annotations

from typing import Any

from sage.all import ChainComplex, matrix


# ---------------------------------------------------------------------------
# Boundary matrix
# ---------------------------------------------------------------------------


def _boundary_matrix(
    module: Any,
    basis_source: list,
    key_to_idx_target: dict,
    n_target: int,
    *,
    sparse: bool,
    strict: bool = False,
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
    strict:
        Currently unused; kept for API compatibility.

    Returns
    -------
    A SageMath matrix of size ``n_target × len(basis_source)`` over the
    base ring of *module*.
    """
    base_ring = module.base_ring()
    n_source = len(basis_source)
    M = matrix(base_ring, n_target, n_source, sparse=sparse)
    for j, elem in enumerate(basis_source):
        bdry = module.boundary(elem)
        for key, coeff in bdry:
            i = key_to_idx_target.get(key)
            if i is not None:
                M[i, j] += coeff
            else:
                raise ValueError(
                    f"Boundary of basis element {elem} contains key {key} "
                    "not found in target basis keys"
                )
    return M


def _get_basis_elements(module: Any, d: int, weight: int | None = None) -> tuple[list, list]:
    """Return (basis_elements, basis_keys).

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
    elems: list = []
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
    strict: bool = False,
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
    strict:
        Currently unused; kept for API compatibility.

    Returns
    -------
    A SageMath :class:`~sage.homology.chain_complex.ChainComplex` with
    ``degree_of_differential=-1``.

    Examples
    --------
    >>> from sage.all import QQ
    >>> from uconf import Surjection
    >>> from uconf.homology import chain_complex
    >>> S2 = Surjection(2, QQ)
    >>> C = chain_complex(S2, degrees=range(4))
    >>> C.homology()  # doctest: +SKIP
    """
    base_ring = module.base_ring()

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

    for d in extended_degrees:
        basis_elems, basis_keys = _get_basis_elements(module, d, weight)
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
            source,
            key_to_idx[d - 1],
            n_target,
            sparse=sparse,
            strict=strict,
        )

    return ChainComplex(differentials, base_ring=base_ring, degree_of_differential=-1, check=check)


def compute_homology_representatives(module: Any, degree: int, weight: int | None, cc) -> list:
    """
    Compute homology representatives for a given module at a specific degree and weight.

    Extracts basis elements from the module and uses the chain complex homology computation
    to generate representative elements. Each homology generator is converted back to a
    module element by expressing it as a linear combination of basis elements.

    Args:
        module (Any): The module for which to compute homology representatives.
        degree (int): The homological degree at which to compute representatives.
        weight (int | None): The weight parameter for basis element selection, or None if not applicable.
        cc: The chain complex object with a homology method that accepts degree and generators=True.

    Returns:
        list: A list of module elements representing the homology generators at the given degree.
    """
    basis_elems, _ = _get_basis_elements(module, degree, weight)

    ho = cc.homology(degree, generators=True)
    # The method returns a list of pairs (vector space, generator), but if there are no generators it returns `(Vector space of dimension 0 over Rational Field, ())` instead of an empty list, so we check for that case and return an empty list of representatives.
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
    strict: bool = False,
) -> list:
    """Return cycle representatives for a basis of the homology in *degree*.

    Parameters
    ----------
    module:
        A dg-module (same requirements as :func:`chain_complex`).
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
        Passed through to :func:`chain_complex`.  See its documentation.
    strict:
        Passed through to :func:`compute_chain_complex`.  Verifies that
        non-canonical boundary keys have canonical representatives in the
        target basis.

    Returns
    -------
    A list of elements of *module* that are cycles (``boundary(x) == 0``)
    and whose homology classes form a basis of ``H_degree(module)``.

    Examples
    --------
    >>> from sage.all import QQ
    >>> from uconf import Surjection
    >>> from uconf.homology import homology_basis
    >>> S2 = Surjection(2, QQ)
    >>> homology_basis(S2, 0)  # doctest: +SKIP
    """
    if degrees is None:
        degrees = range(degree - 1, degree + 2)
    else:
        if degree not in degrees:
            raise ValueError(f"degree {degree} must be contained in the supplied range {degrees}")

    C = compute_chain_complex(module, degrees, weight=weight, strict=strict)

    return compute_homology_representatives(module, degree, weight, C)
