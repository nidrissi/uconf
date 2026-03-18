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

from sage.all import ChainComplex, Family, matrix


def _boundary_matrix(
    module: Any,
    basis_source: list,
    key_to_idx_target: dict,
    n_target: int,
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
    n_source = len(basis_source)
    M = matrix(base_ring, n_target, n_source)
    for j, elem in enumerate(basis_source):
        bdry = module.boundary(elem)
        for key, coeff in bdry:
            i = key_to_idx_target.get(key)
            if i is not None:
                M[i, j] += coeff
    return M


def _deduplicated_basis(module: Any, d: int) -> tuple[list, list]:
    """Return (basis_elements, basis_keys) with duplicate keys removed."""
    seen: set = set()
    elems: list = []
    keys: list = []
    for b in module.graded_basis(d):
        key = list(b.support())[0]
        if key not in seen:
            seen.add(key)
            elems.append(b)
            keys.append(key)
    return elems, keys


def chain_complex(module: Any, degrees: range) -> Any:
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
        A :class:`range` of integer degrees to include.  The resulting chain
        complex will have modules in each of these degrees (possibly
        zero-dimensional) and differentials between consecutive degrees.

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

    # Collect basis elements and keys for each degree (deduplicated)
    basis_by_degree: dict[int, list] = {}
    keys_by_degree: dict[int, list] = {}
    key_to_idx: dict[int, dict] = {}

    for d in degrees:
        basis_elems, basis_keys = _deduplicated_basis(module, d)
        basis_by_degree[d] = basis_elems
        keys_by_degree[d] = basis_keys
        key_to_idx[d] = {k: i for i, k in enumerate(basis_keys)}

    # Build differential matrices: d_n : C_n -> C_{n-1}
    differentials: dict[int, Any] = {}
    for d in degrees:
        if d - 1 not in key_to_idx:
            # Target degree not in range; if there are source basis elements,
            # we still need a zero matrix so the complex knows the rank of C_d.
            if basis_by_degree[d]:
                differentials[d] = matrix(
                    base_ring, 0, len(basis_by_degree[d])
                )
            continue
        n_target = len(keys_by_degree[d - 1])
        source = basis_by_degree[d]
        if not source and n_target == 0:
            continue
        differentials[d] = _boundary_matrix(
            module, source, key_to_idx[d - 1], n_target
        )

    return ChainComplex(differentials, base_ring=base_ring, degree_of_differential=-1)


def homology_basis(module: Any, degree: int, *, degrees: range | None = None) -> list:
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
        ``range(degree - 1, degree + 2)`` is used.  Pass a wider range
        if the module has non-trivial basis below ``degree - 1``.

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
        lo = max(degree - 1, 0)
        degrees = range(lo, degree + 2)
    else:
        if degree not in degrees:
            raise ValueError(
                f"degree {degree} must be contained in the supplied range {degrees}"
            )

    C = chain_complex(module, degrees)

    # Retrieve the (deduplicated) basis elements in the requested degree
    basis_elems, _ = _deduplicated_basis(module, degree)

    h = C.homology(generators=True)
    result: list = []
    for _vspace, gen in h.get(degree, []):
        vec = gen.vector(degree)
        elem = module.zero()
        for i, coeff in enumerate(vec):
            if coeff:
                elem += coeff * basis_elems[i]
        result.append(elem)

    return result
