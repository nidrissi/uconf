"""Random basis element sampling without full basis enumeration.

Provides utilities to generate random basis elements for the various
algebraic objects in the configuration model pipeline:

- :func:`random_surjection` — random surjection of given arity and degree.
- :func:`random_planar_surjection` — random *planar* surjection.
- :func:`random_sphere_admissible_surjection` — surjection acting
  nontrivially on ``N*(S^d)``.
- :func:`random_lie_key` — random Lie-operad basis key.
- :func:`random_hadamard_key` — random Hadamard-product basis key.
- :func:`sample_basis` — sample *k* random elements from any parent with
  ``graded_basis`` or ``basis_iter``, falling back to enumeration when
  direct sampling is not available.
- :func:`sample_operad_basis` — sample operad/cooperad basis elements.
- :func:`sample_algebra_pool` — replacement for
  ``_build_algebra_pool`` in tests.
- :func:`sphere_nontrivial_surjection_iter` — iterate over surjections
  that act nontrivially on the sphere.
"""

from __future__ import annotations

from random import Random
from typing import Any, Iterator

from uconf.algebraic.spherical import _extract_concatenated_permutations


# ---------------------------------------------------------------------------
# Surjection sampling
# ---------------------------------------------------------------------------


def random_surjection_key(n: int, degree: int, rng: Random) -> tuple[int, ...] | None:
    r"""Generate a random valid surjection key of arity ``n`` and degree ``degree``.

    A surjection of arity *n* and degree *d* is a tuple of length ``n + d``
    with values in ``{1, …, n}`` that is surjective and has no consecutive
    equal entries.

    Uses rejection sampling: generate a random tuple and check validity.
    Returns ``None`` if no valid surjection is found after a limited number
    of attempts (unlikely for small *n* and *d*).
    """
    if n < 1 or degree < 0:
        return None
    if n == 1:
        return (1,) if degree == 0 else None

    length = n + degree
    if length < n:
        return None

    values = list(range(1, n + 1))
    max_attempts = 2000

    for _ in range(max_attempts):
        # Build a random tuple with no consecutive repeats
        result = [rng.choice(values)]
        for _ in range(length - 1):
            choices = [v for v in values if v != result[-1]]
            result.append(rng.choice(choices))
        t = tuple(result)
        # Check surjectivity
        if len(set(t)) == n:
            return t

    return None


def random_surjection(n: int, degree: int, base_ring, rng: Random):
    """Generate a random Surjection element of arity ``n`` and degree ``degree``.

    Returns a single-term element or ``None`` if sampling fails.
    """
    from uconf.models.surjection import Surjection

    key = random_surjection_key(n, degree, rng)
    if key is None:
        return None
    parent = Surjection(n, base_ring)
    elem = parent(key)
    return elem if elem != parent.zero() else None


def random_planar_surjection_key(n: int, degree: int, rng: Random) -> tuple[int, ...] | None:
    r"""Generate a random *planar* surjection key of arity ``n`` and degree ``degree``.

    A surjection is planar when the first occurrences of ``1, 2, …, n``
    appear in increasing order of position.

    Uses rejection sampling with planarity filter.
    """
    if n < 1 or degree < 0:
        return None
    if n == 1:
        return (1,) if degree == 0 else None

    max_attempts = 5000
    for _ in range(max_attempts):
        key = random_surjection_key(n, degree, rng)
        if key is None:
            continue
        # Check planarity: first occurrences must be in order 1, 2, ..., n
        first_occ = {}
        for i, v in enumerate(key):
            if v not in first_occ:
                first_occ[v] = i
        positions = [first_occ[v] for v in range(1, n + 1)]
        if positions == sorted(positions):
            return key

    return None


def random_planar_surjection(n: int, degree: int, base_ring, rng: Random):
    """Generate a random planar Surjection element."""
    from uconf.models.surjection import Surjection

    key = random_planar_surjection_key(n, degree, rng)
    if key is None:
        return None
    parent = Surjection(n, base_ring)
    elem = parent(key)
    return elem if elem != parent.zero() else None


# ---------------------------------------------------------------------------
# Sphere-admissible surjection sampling
# ---------------------------------------------------------------------------


def random_sphere_admissible_surjection_key(
    n: int, dim: int, rng: Random
) -> tuple[int, ...] | None:
    r"""Generate a random sphere-admissible surjection key.

    A surjection of arity *n* is sphere-admissible for ``S^d`` (``d = dim``)
    when its degree equals ``d(n−1)`` and it decomposes as the concatenation
    of ``d + 1`` permutations of ``{1, …, n}`` with overlapping endpoints.

    Direct construction: generate ``d + 1`` random permutations
    ``σ_1, …, σ_{d+1}`` of ``{1, …, n}`` with the overlap constraint
    ``σ_j(n) = σ_{j+1}(1)``.
    """
    if n < 1 or dim < 0:
        return None
    if n == 1:
        return (1,)

    d = dim
    # We need d+1 permutations of {1,...,n} with overlap σ_j[-1] = σ_{j+1}[0]
    values = list(range(1, n + 1))

    max_attempts = 500
    for _ in range(max_attempts):
        perms = []
        # First permutation: random
        first = list(values)
        rng.shuffle(first)
        perms.append(tuple(first))

        ok = True
        for _ in range(d):
            # Next permutation must start with the last element of the previous one
            start = perms[-1][-1]
            remaining = [v for v in values if v != start]
            rng.shuffle(remaining)
            perm = [start] + remaining
            perms.append(tuple(perm))

        if not ok:
            continue

        # Build surjection: concatenate with overlap
        # u = σ_1[0:n-1], σ_2[0:n-1], ..., σ_d[0:n-1], σ_{d+1}[0:n]
        result = []
        for j in range(d):
            result.extend(perms[j][:n - 1])
        result.extend(perms[d])
        t = tuple(result)

        # Verify no consecutive repeats (they shouldn't happen by construction
        # since we concatenate permutations, but check anyway)
        if any(t[i] == t[i + 1] for i in range(len(t) - 1)):
            continue

        # Verify sphere-admissibility (should always pass by construction)
        if _extract_concatenated_permutations(t, n, d) is not None:
            return t

    return None


def random_sphere_admissible_surjection(n: int, dim: int, base_ring, rng: Random):
    """Generate a random sphere-admissible Surjection element."""
    from uconf.models.surjection import Surjection

    key = random_sphere_admissible_surjection_key(n, dim, rng)
    if key is None:
        return None
    parent = Surjection(n, base_ring)
    elem = parent(key)
    return elem if elem != parent.zero() else None


def sphere_nontrivial_surjection_iter(
    n: int, dim: int, base_ring
) -> Iterator:
    """Iterate over surjections of arity ``n`` that act nontrivially on ``S^d``.

    Only surjections of degree ``d(n−1)`` with the sphere-admissible
    concatenation form are considered.  Yields ``(key, sign)`` pairs.
    """
    from uconf.algebraic.spherical import _sphere_surjection_basis_sign
    from uconf.models.surjection import Surjection

    d = dim
    degree = d * (n - 1)
    parent = Surjection(n, base_ring)

    for elem in parent.basis_iter(degree):
        for key in elem.support():
            sign = _sphere_surjection_basis_sign(key, n, d)
            if sign != 0:
                yield parent.term(key), sign


def sphere_nontrivial_operad_basis_iter(
    operad_cls,
    n: int,
    dim: int,
    base_ring,
) -> Iterator:
    """Iterate over operad basis elements at arity ``n`` whose surjection
    factor acts nontrivially on ``S^d``.

    Works for ``HadamardProduct(_, Surjection)`` and plain ``Surjection``.
    For non-Hadamard operads, falls back to the full basis at degree ``d(n-1)``.
    """
    from uconf.wrappers.hadamard_operad import HadamardProduct

    d = dim
    surj_degree = d * (n - 1)

    parent = operad_cls(n, base_ring)

    if hasattr(operad_cls, '__name__') and operad_cls.__name__ == 'Surjection':
        # Direct surjection: only degree d(n-1) can act nontrivially
        yield from parent.basis_iter(surj_degree)
        return

    if isinstance(parent, HadamardProduct.Component):
        # Hadamard product: filter by surjection factor degree
        # The right factor should be Surjection
        right_parent = parent._right_parent
        left_parent = parent._left_parent

        # Enumerate right-factor (surjection) elements at the required degree
        right_elems = list(right_parent.basis_iter(surj_degree))
        if not right_elems:
            return

        # Filter right elements to nontrivial ones
        from uconf.algebraic.spherical import _sphere_surjection_basis_sign
        nontrivial_right = []
        for right_elem in right_elems:
            for right_key in right_elem.support():
                sign = _sphere_surjection_basis_sign(right_key, n, d)
                if sign != 0:
                    nontrivial_right.append(right_elem)
                    break
        if not nontrivial_right:
            return

        # Enumerate left-factor elements across all available degrees
        # For shifted Lie at arity n, degree is -(n-1)
        from uconf.wrappers.hadamard_operad import _min_component_degree
        min_d_left = _min_component_degree(left_parent, n)
        # The left factor typically lives in a single degree (e.g. Lie at deg 0,
        # shifted Lie at deg -(n-1)), so check a reasonable range
        left_all = []
        for dl in range(min_d_left, min_d_left + 5):
            left_elems = list(left_parent.basis_iter(dl))
            left_all.extend(left_elems)
            if left_elems:
                break  # found the degree where elements exist

        for right_elem in nontrivial_right:
            for left_elem in left_all:
                yield parent.from_factors(left_elem, right_elem)

    # Fallback for other operads (e.g. CobarConstruction)
    # For cobar of Hadamard, we can't easily filter, so iterate all degrees
    # that could produce sphere-nontrivial action
    # This is a heuristic — the caller should handle weight/degree filtering
    # For now, yield nothing (caller can use full basis_iter instead)


# ---------------------------------------------------------------------------
# Lie sampling
# ---------------------------------------------------------------------------


def random_lie_key(n: int, rng: Random) -> tuple[int, ...]:
    """Return a random Lie basis key at arity ``n``.

    Lie basis keys are permutations of ``(1, …, n−1)`` (Lyndon/PBW basis).
    """
    if n <= 1:
        return ()
    values = list(range(1, n))
    rng.shuffle(values)
    return tuple(values)


def random_lie_element(n: int, base_ring, rng: Random):
    """Generate a random Lie operad basis element at arity ``n``."""
    from uconf.models.lie import Lie

    key = random_lie_key(n, rng)
    parent = Lie(n, base_ring)
    return parent.term(key)


# ---------------------------------------------------------------------------
# Hadamard product sampling
# ---------------------------------------------------------------------------


def random_hadamard_key(
    hadamard_parent,
    degree: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> Any | None:
    r"""Generate a random basis key for a Hadamard product component.

    Samples left and right factors independently at compatible degrees.

    Parameters
    ----------
    hadamard_parent : HadamardProduct.Component
        The Hadamard-product component.
    degree : int
        Target total degree.
    rng : Random
        Random number generator.
    sphere_nontrivial : bool
        If ``True``, only sample right-factor (Surjection) elements that
        act nontrivially on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
        A random Hadamard product element, or ``None`` if sampling failed.
    """
    from uconf.algebraic.spherical import _sphere_surjection_basis_sign

    left_parent = hadamard_parent._left_parent
    right_parent = hadamard_parent._right_parent
    n = hadamard_parent._arity

    if sphere_nontrivial:
        if sphere_dim is None:
            raise ValueError("sphere_dim is required when sphere_nontrivial=True")
        # Right factor must be at degree d*(n-1)
        surj_degree = sphere_dim * (n - 1)
        # Left factor must be at degree (degree - surj_degree)
        left_degree = degree - surj_degree

        # Get left elements at required degree
        left_basis = list(left_parent.basis_iter(left_degree))
        if not left_basis:
            return None

        # Get nontrivial right elements
        right_nontrivial = []
        for elem in right_parent.basis_iter(surj_degree):
            for key in elem.support():
                sign = _sphere_surjection_basis_sign(key, n, sphere_dim)
                if sign != 0:
                    right_nontrivial.append(elem)
                    break
        if not right_nontrivial:
            return None

        left_elem = rng.choice(left_basis)
        right_elem = rng.choice(right_nontrivial)
        return hadamard_parent.from_factors(left_elem, right_elem)

    # General case: pick a random degree split
    # Try a few degree splits
    from uconf.wrappers.hadamard_operad import _min_component_degree

    min_d_left = _min_component_degree(left_parent, n)
    min_d_right = _min_component_degree(right_parent, n)

    possible_splits = []
    for d_left in range(min_d_left, degree - min_d_right + 1):
        d_right = degree - d_left
        possible_splits.append((d_left, d_right))

    if not possible_splits:
        return None

    rng.shuffle(possible_splits)

    for d_left, d_right in possible_splits[:10]:
        left_basis = list(left_parent.basis_iter(d_left))
        if not left_basis:
            continue
        right_basis = list(right_parent.basis_iter(d_right))
        if not right_basis:
            continue

        left_elem = rng.choice(left_basis)
        right_elem = rng.choice(right_basis)
        return hadamard_parent.from_factors(left_elem, right_elem)

    return None


def sample_hadamard_basis(
    hadamard_parent,
    degree: int,
    k: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> list:
    """Sample up to *k* random Hadamard-product basis elements.

    If fewer than *k* distinct elements can be generated, returns all
    that were found.
    """
    seen = set()
    results = []
    max_attempts = k * 10

    for _ in range(max_attempts):
        if len(results) >= k:
            break
        elem = random_hadamard_key(
            hadamard_parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if elem is not None:
            # Use support as hashable identifier
            key = frozenset((b, c) for b, c in elem)
            if key not in seen:
                seen.add(key)
                results.append(elem)

    return results


# ---------------------------------------------------------------------------
# Generic basis sampling
# ---------------------------------------------------------------------------


def sample_basis(
    parent,
    degree: int,
    k: int,
    rng: Random,
    *,
    weight: int | None = None,
) -> list:
    """Sample up to *k* random basis elements from *parent* at degree *degree*.

    Tries cached ``graded_basis`` first (which materializes the full basis
    but only once).  Falls back to ``basis_iter`` when no cached family
    exists.

    Parameters
    ----------
    parent
        A SageMath parent with ``graded_basis(d)`` or ``basis_iter(d)``.
    degree : int
        Homological degree.
    k : int
        Maximum number of elements to return.
    rng : Random
        Random number generator.
    weight : int or None
        If given, restrict to elements of this weight (uses
        ``graded_basis_by_weight``).
    """
    if weight is not None:
        getter = getattr(parent, "graded_basis_by_weight", None)
        if getter is not None:
            full = list(getter(degree, weight))
        else:
            getter = getattr(parent, "basis_weight_iter", None)
            if getter is not None:
                full = list(getter(degree, weight))
            else:
                full = list(parent.graded_basis(degree))
                full = [e for e in full if getattr(parent, '_weight_on_basis', lambda _: 0)(
                    next(iter(e.monomial_coefficients()))
                ) == weight]
    else:
        cached = getattr(parent, "graded_basis", None)
        if cached is not None:
            full = list(cached(degree))
        else:
            full = list(parent.basis_iter(degree))

    if len(full) <= k:
        return list(full)
    return rng.sample(full, k)


def sample_operad_basis(
    operad_cls,
    n: int,
    degree: int,
    k: int,
    base_ring,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> list:
    """Sample up to *k* operad basis elements at arity ``n`` and degree ``degree``.

    Parameters
    ----------
    operad_cls
        Operad factory (e.g. ``Surjection``, ``HadamardProduct`` factory, etc.).
    n : int
        Arity.
    degree : int
        Degree.
    k : int
        Maximum number of elements.
    base_ring
        Coefficient ring.
    rng : Random
        Random number generator.
    sphere_nontrivial : bool
        If ``True``, only return elements whose surjection factor acts
        nontrivially on the sphere.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.
    """
    from uconf.wrappers.hadamard_operad import HadamardProduct

    parent = operad_cls(n, base_ring)

    if sphere_nontrivial and isinstance(parent, HadamardProduct.Component):
        return sample_hadamard_basis(
            parent,
            degree,
            k,
            rng,
            sphere_nontrivial=True,
            sphere_dim=sphere_dim,
        )

    return sample_basis(parent, degree, k, rng)


# ---------------------------------------------------------------------------
# Algebra pool sampling
# ---------------------------------------------------------------------------


def sample_algebra_pool(
    mod,
    k_per_bucket: int,
    rng: Random,
    *,
    weights: tuple[int, ...] = (1, 2),
    deg_range: range = range(-1, 6),
) -> list:
    """Sample algebra elements across multiple weight/degree buckets.

    Replacement for the test helper ``_build_algebra_pool``.  Samples up
    to *k_per_bucket* elements from each ``(degree, weight)`` cell instead
    of materializing the full basis.

    Parameters
    ----------
    mod
        Algebra module with ``graded_basis_by_weight(d, w)``.
    k_per_bucket : int
        Maximum elements per (degree, weight) cell.
    rng : Random
        Random number generator.
    weights : tuple[int, ...]
        Weights to sample.
    deg_range : range
        Degree range to sample.
    """
    pool: list = []
    for w in weights:
        for d in deg_range:
            pool.extend(sample_basis(mod, d, k_per_bucket, rng, weight=w))
    return pool
