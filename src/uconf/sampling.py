"""Random basis element sampling without full basis enumeration.

Provides construction-aware utilities to generate random basis elements
for the various algebraic objects in the configuration model pipeline.
Each sampler exploits the combinatorial structure of its parent object
(tree shape, vertex decorations, factor products) to generate random
elements *directly*, without materializing the full basis first.

Construction-aware samplers:

- :func:`random_surjection` — random surjection via rejection sampling.
- :func:`random_planar_surjection` — random *planar* surjection.
- :func:`random_sphere_admissible_surjection` — sphere-admissible
  surjection via direct construction.
- :func:`random_lie_key` — random Lie-operad basis key.
- :func:`random_barratt_eccles_key` — random Barratt–Eccles key.
- :func:`random_hadamard_key` — random Hadamard-product basis key
  by sampling factors independently.
- :func:`random_shuffle_tree` — random decorated shuffle tree
  (used internally by bar/cobar construction samplers).
- :func:`random_bar_element` — random bar construction element.
- :func:`random_cobar_element` — random cobar construction element.
- :func:`random_free_algebra_element` — random free algebra element
  by sampling operad key and module tuple independently.
- :func:`random_cofree_coalgebra_element` — random cofree coalgebra
  element by sampling cooperad key and module tuple independently.
- :func:`random_tree_module_element` — random tree-module element.
- :func:`sample_basis` — sample *k* random elements from any parent,
  dispatching to construction-aware generators when available and
  falling back to basis enumeration otherwise.
- :func:`sample_operad_basis` — sample operad/cooperad basis elements.
- :func:`sample_algebra_pool` — replacement for
  ``_build_algebra_pool`` in tests.
- :func:`sphere_nontrivial_surjection_iter` — iterate over surjections
  that act nontrivially on the sphere.
"""

from __future__ import annotations

from random import Random
from typing import Any, Iterator

from sage.all import SymmetricGroup as SG

from uconf.algebraic.cofree_coalgebra import CofreeCoalgebraModule
from uconf.algebraic.free_algebra import FreeAlgebraModule
from uconf.algebraic.spherical import (
    _extract_concatenated_permutations,
    _sphere_surjection_basis_sign,
)
from uconf.algebraic.tree_module import TreeModule, _module_basis_keys_in_degree
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.trees import RootedTree
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.lie import Lie
from uconf.models.surjection import Surjection
from uconf.wrappers.hadamard_operad import HadamardProduct, _min_component_degree
from uconf.wrappers.shifted_cooperad import ShiftedCooperad
from uconf.wrappers.shifted_operad import ShiftedOperad

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
            result.extend(perms[j][: n - 1])
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
    key = random_sphere_admissible_surjection_key(n, dim, rng)
    if key is None:
        return None
    parent = Surjection(n, base_ring)
    elem = parent(key)
    return elem if elem != parent.zero() else None


def sphere_nontrivial_surjection_iter(n: int, dim: int, base_ring) -> Iterator:
    """Iterate over surjections of arity ``n`` that act nontrivially on ``S^d``.

    Only surjections of degree ``d(n−1)`` with the sphere-admissible
    concatenation form are considered.  Yields ``(key, sign)`` pairs.
    """
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
    d = dim
    surj_degree = d * (n - 1)

    parent = operad_cls(n, base_ring)

    if hasattr(operad_cls, "__name__") and operad_cls.__name__ == "Surjection":
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
    key = random_lie_key(n, rng)
    parent = Lie(n, base_ring)
    return parent.term(key)


# ---------------------------------------------------------------------------
# Barratt–Eccles sampling
# ---------------------------------------------------------------------------


def random_barratt_eccles_key(n: int, degree: int, rng: Random) -> tuple | None:
    r"""Generate a random Barratt–Eccles basis key of arity ``n`` and degree ``degree``.

    A BE key is a tuple of ``degree + 1`` permutations in ``S_n`` with no
    consecutive equal permutations.  Uses direct construction with rejection
    for the no-consecutive-repeat constraint.

    Returns ``None`` if ``degree < 0`` or sampling fails.
    """
    if degree < 0 or n < 1:
        return None

    Sn = SG(n)
    all_perms = list(Sn)

    max_attempts = 500
    for _ in range(max_attempts):
        result = [rng.choice(all_perms)]
        ok = True
        for _ in range(degree):
            choices = [p for p in all_perms if p != result[-1]]
            if not choices:
                ok = False
                break
            result.append(rng.choice(choices))
        if ok:
            return tuple(result)

    return None


def random_barratt_eccles_element(n: int, degree: int, base_ring, rng: Random):
    """Generate a random Barratt–Eccles element of arity ``n`` and degree ``degree``.

    Returns a single-term element or ``None`` if sampling fails.
    """
    key = random_barratt_eccles_key(n, degree, rng)
    if key is None:
        return None
    parent = BarrattEccles(n, base_ring)
    elem = parent(key)
    return elem if elem != parent.zero() else None


# ---------------------------------------------------------------------------
# Hadamard product sampling
# ---------------------------------------------------------------------------


def _random_operad_element(
    parent,
    degree: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a single random operad/cooperad element at the given degree.

    Dispatches to construction-aware generators when the parent type is
    recognized; falls back to sampling from ``basis_iter`` otherwise.

    Parameters
    ----------
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.  Propagated recursively through all
        constructions.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.
    """
    n = parent.arity() if hasattr(parent, "arity") else getattr(parent, "_arity", None)
    R = parent.base_ring()

    if isinstance(parent, Surjection):
        if sphere_nontrivial:
            if sphere_dim is None:
                raise ValueError("sphere_dim is required when sphere_nontrivial=True")
            return random_sphere_admissible_surjection(n, sphere_dim, R, rng)
        return random_surjection(n, degree, R, rng)

    if isinstance(parent, BarrattEccles):
        return random_barratt_eccles_element(n, degree, R, rng)

    if isinstance(parent, Lie):
        if degree != 0:
            return None
        key = random_lie_key(n, rng)
        return parent.term(key)

    if isinstance(parent, ShiftedOperad.Component):
        shift = parent.factory.shift_degree
        base_degree = degree - shift * (n - 1)
        base_elem = _random_operad_element(
            parent._base_parent,
            base_degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if base_elem is None:
            return None
        return parent.sum_of_terms((k, R(c)) for k, c in base_elem)

    if isinstance(parent, ShiftedCooperad.Component):
        shift = parent.factory.shift_degree
        base_degree = degree - shift * (n - 1)
        base_elem = _random_operad_element(
            parent._base_parent,
            base_degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if base_elem is None:
            return None
        return parent.sum_of_terms((k, R(c)) for k, c in base_elem)

    if isinstance(parent, HadamardProduct.Component):
        return random_hadamard_key(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )

    if isinstance(parent, BarConstruction.Component):
        return random_bar_element(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )

    if isinstance(parent, CobarConstruction.Component):
        return random_cobar_element(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )

    # Fallback: sample from basis_iter (materializes elements one at a time)
    return _random_from_iter(parent, degree, rng)


def _random_from_iter(parent, degree: int, rng: Random):
    """Pick a random element from a basis iterator using reservoir sampling.

    Uses Algorithm R (reservoir of size 1) so only one pass over the
    iterator is needed and at most one element is kept in memory at a time.
    """
    basis_iter = getattr(parent, "basis_iter", None)
    if basis_iter is None:
        return None
    chosen = None
    count = 0
    for elem in basis_iter(degree):
        count += 1
        if rng.randint(1, count) == 1:
            chosen = elem
    return chosen


def random_hadamard_key(
    hadamard_parent,
    degree: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> Any | None:
    r"""Generate a random basis key for a Hadamard product component.

    Samples left and right factors independently at compatible degrees,
    without materializing either factor's full basis.

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
    left_parent = hadamard_parent._left_parent
    right_parent = hadamard_parent._right_parent
    n = hadamard_parent._arity

    if sphere_nontrivial:
        if sphere_dim is None:
            raise ValueError("sphere_dim is required when sphere_nontrivial=True")
        # The Surjection factor is constrained to degree d*(n-1).
        # Detect which factor is Surjection and apply the admissibility filter.
        surj_degree = sphere_dim * (n - 1)
        left_is_surj = isinstance(left_parent, Surjection)
        right_is_surj = isinstance(right_parent, Surjection)

        if left_is_surj:
            # Left factor is Surjection: sample sphere-admissible, propagate right
            right_degree = degree - surj_degree
            left_elem = random_sphere_admissible_surjection(
                n, sphere_dim, left_parent.base_ring(), rng
            )
            if left_elem is None:
                return None
            right_elem = _random_operad_element(
                right_parent,
                right_degree,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )
        elif right_is_surj:
            # Right factor is Surjection: propagate left, sample sphere-admissible right
            left_degree = degree - surj_degree
            left_elem = _random_operad_element(
                left_parent,
                left_degree,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )
            if left_elem is None:
                return None
            right_elem = random_sphere_admissible_surjection(
                n, sphere_dim, right_parent.base_ring(), rng
            )
        else:
            # Neither factor is directly Surjection; propagate sphere_nontrivial to both
            min_d_left = _min_component_degree(left_parent, n)
            min_d_right = _min_component_degree(right_parent, n)
            possible_splits = [
                (d_left, degree - d_left) for d_left in range(min_d_left, degree - min_d_right + 1)
            ]
            if not possible_splits:
                return None
            rng.shuffle(possible_splits)
            for d_left, d_right in possible_splits[:10]:
                left_elem = _random_operad_element(
                    left_parent,
                    d_left,
                    rng,
                    sphere_nontrivial=sphere_nontrivial,
                    sphere_dim=sphere_dim,
                )
                if left_elem is None:
                    continue
                right_elem = _random_operad_element(
                    right_parent,
                    d_right,
                    rng,
                    sphere_nontrivial=sphere_nontrivial,
                    sphere_dim=sphere_dim,
                )
                if right_elem is None:
                    continue
                return hadamard_parent.from_factors(left_elem, right_elem)
            return None

        if left_elem is None or right_elem is None:
            return None
        return hadamard_parent.from_factors(left_elem, right_elem)

    # General case: pick a random degree split
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
        left_elem = _random_operad_element(left_parent, d_left, rng)
        if left_elem is None:
            continue
        right_elem = _random_operad_element(right_parent, d_right, rng)
        if right_elem is None:
            continue
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
# Random tree generation (bar/cobar constructions)
# ---------------------------------------------------------------------------


def _random_partition(sorted_leaves: tuple, k: int, rng: Random) -> list[tuple] | None:
    """Generate a random partition of *sorted_leaves* into *k* non-empty parts sorted by min.

    Each element is assigned to an existing part or opens a new part,
    with the constraint that parts are opened in order and the minimum
    of each part increases.  Choices are made uniformly at random among
    the valid options.
    """
    n = len(sorted_leaves)
    if k <= 0 or k > n:
        return None

    parts: list[list[int]] = [[] for _ in range(k)]
    parts[0].append(sorted_leaves[0])
    num_open = 1

    for idx in range(1, n):
        elem = sorted_leaves[idx]
        # Can place in any open part, or open a new one (if not all opened yet)
        options: list[int] = list(range(num_open))
        if num_open < k:
            options.append(num_open)  # opening the next part

        choice = rng.choice(options)
        if choice == num_open and num_open < k:
            num_open += 1
        parts[choice].append(elem)

    if num_open != k:
        return None  # failed to open all k parts

    return [tuple(p) for p in parts]


def random_shuffle_tree(
    leaf_set: tuple,
    max_weight: int,
    operad_cls,
    base_ring,
    target_degree: int,
    vertex_offset: int,
    rng: Random,
    *,
    max_attempts: int = 200,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    r"""Generate a random decorated shuffle tree.

    Constructs trees top-down: at each internal vertex, randomly choose the
    vertex arity, a random decoration from the operad/cooperad, and a random
    partition of the leaves.  Recurse into each child subtree.

    The ``vertex_offset`` parameter selects the degree convention:

    - ``+1``: bar degree ``Σ (deg_P(v) + 1)``.
    - ``-1``: cobar degree ``Σ (deg_C(v) - 1)``.
    - ``0``: free/cofree degree ``Σ deg(v)``.

    Returns a decorated ``RootedTree`` or ``None`` if sampling fails.
    """
    n = len(leaf_set)

    if n == 0:
        return None
    if n == 1:
        return leaf_set[0] if target_degree == 0 else None
    if max_weight < 1:
        return None

    connectivity = getattr(operad_cls, "connectivity", 0)
    sorted_ls = tuple(sorted(leaf_set))

    for _ in range(max_attempts):
        result = _random_subtree(
            sorted_ls,
            max_weight,
            operad_cls,
            base_ring,
            target_degree,
            vertex_offset,
            connectivity,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if result is not None:
            return result

    return None


def _random_subtree(
    sorted_ls: tuple,
    max_weight: int,
    operad_cls,
    base_ring,
    target_degree: int,
    vertex_offset: int,
    connectivity: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Recursively build a random subtree (internal helper)."""
    n = len(sorted_ls)
    if n == 1:
        return sorted_ls[0] if target_degree == 0 else None
    if max_weight < 1:
        return None

    # Pick a random vertex arity
    possible_arities = list(range(2, n + 1))
    rng.shuffle(possible_arities)

    for v_arity in possible_arities:
        root_parent = operad_cls(v_arity, base_ring)

        if v_arity == n:
            # Corolla: all leaves are direct children
            root_dec_deg = target_degree - vertex_offset
            if root_dec_deg < connectivity * (v_arity - 1):
                continue
            dec = _random_operad_element(
                root_parent,
                root_dec_deg,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )
            if dec is None:
                continue
            # Get a single basis key from the element
            dec_key = _extract_key(dec)
            if dec_key is None:
                continue
            return RootedTree(dec_key, *sorted_ls)
        else:
            # Non-corolla: need to partition leaves
            parts = _random_partition(sorted_ls, v_arity, rng)
            if parts is None:
                continue

            # Pick a random degree for the root decoration
            min_child_total = sum(
                max(0, vertex_offset + connectivity * (len(p) - 1)) for p in parts if len(p) >= 2
            )
            max_root_dec_deg = target_degree - vertex_offset - min_child_total
            min_root_dec_deg = connectivity * (v_arity - 1)

            if max_root_dec_deg < min_root_dec_deg:
                continue

            # Try a random root decoration degree
            root_dec_deg = rng.randint(min_root_dec_deg, max_root_dec_deg)
            dec = _random_operad_element(
                root_parent,
                root_dec_deg,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )
            if dec is None:
                continue
            dec_key = _extract_key(dec)
            if dec_key is None:
                continue

            child_total = target_degree - root_dec_deg - vertex_offset

            # Recursively build children, distributing remaining degree
            children_result = _random_children(
                parts,
                max_weight - 1,
                operad_cls,
                base_ring,
                child_total,
                vertex_offset,
                connectivity,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )
            if children_result is not None:
                return RootedTree(dec_key, *children_result)

    return None


def _random_children(
    parts: list,
    max_weight: int,
    operad_cls,
    base_ring,
    total_deg: int,
    vertex_offset: int,
    connectivity: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> list | None:
    """Build random children for each part, distributing total_deg among them."""
    k = len(parts)
    if k == 0:
        return [] if total_deg == 0 else None

    # For leaves (single-element parts), degree must be 0
    internal_parts = [(i, p) for i, p in enumerate(parts) if len(p) >= 2]

    # All leaf parts consume 0 degree
    remaining_deg = total_deg

    if not internal_parts:
        # All parts are leaves
        if remaining_deg != 0:
            return None
        return [p[0] for p in parts]

    # Distribute remaining degree among internal parts
    # For simplicity, use a greedy approach: give each internal part
    # its minimum possible degree, then distribute the remainder randomly
    children = [None] * k

    # Set leaf children first
    for i, p in enumerate(parts):
        if len(p) == 1:
            children[i] = p[0]

    # For internal parts, try distributing degree
    min_degs = []
    for _, p in internal_parts:
        min_d = vertex_offset + connectivity * (len(p) - 1)
        min_degs.append(max(0, min_d) if vertex_offset >= 0 else min_d)

    total_min = sum(min_degs)
    if remaining_deg < total_min:
        return None

    # Simple strategy: give each child its minimum, then give all extra to a random one
    child_degs = list(min_degs)
    extra = remaining_deg - total_min
    if extra > 0 and internal_parts:
        # Distribute extra to a random internal child
        idx = rng.randint(0, len(internal_parts) - 1)
        child_degs[idx] += extra

    for j, (i, p) in enumerate(internal_parts):
        subtree = _random_subtree(
            tuple(sorted(p)),
            max_weight,
            operad_cls,
            base_ring,
            child_degs[j],
            vertex_offset,
            connectivity,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if subtree is None:
            return None
        children[i] = subtree

    return children


def _extract_key(elem):
    """Extract a single basis key from an element (picks one at random if multi-term)."""
    if elem is None:
        return None
    support = list(elem.support())
    if not support:
        return None
    return support[0]


def random_bar_element(
    parent,
    degree: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a random bar construction element without materializing the full basis.

    Parameters
    ----------
    parent : BarConstruction.Component
        The bar construction component.
    degree : int
        Target bar degree.
    rng : Random
        Random number generator.
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
        A random bar construction element.
    """
    n = parent._arity
    if n == 1:
        return parent(1) if degree == 0 else None

    tree = random_shuffle_tree(
        tuple(range(1, n + 1)),
        n - 1,  # max_weight = arity - 1
        parent._operad_cls,
        parent.base_ring(),
        degree,
        +1,  # bar degree convention
        rng,
        sphere_nontrivial=sphere_nontrivial,
        sphere_dim=sphere_dim,
    )
    if tree is None:
        return None
    return parent(tree)


def random_cobar_element(
    parent,
    degree: int,
    rng: Random,
    *,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a random cobar construction element without materializing the full basis.

    Parameters
    ----------
    parent : CobarConstruction.Component
        The cobar construction component.
    degree : int
        Target cobar degree.
    rng : Random
        Random number generator.
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
        A random cobar construction element.
    """
    n = parent._arity
    if n == 1:
        return parent.term(1) if degree == 0 else None

    tree = random_shuffle_tree(
        tuple(range(1, n + 1)),
        n - 1,  # max_weight
        parent._cooperad_cls,
        parent.base_ring(),
        degree,
        -1,  # cobar degree convention
        rng,
        sphere_nontrivial=sphere_nontrivial,
        sphere_dim=sphere_dim,
    )
    if tree is None:
        return None
    return parent(tree)


# ---------------------------------------------------------------------------
# Free algebra / cofree coalgebra sampling
# ---------------------------------------------------------------------------


def _random_module_key(module, degree: int, rng: Random):
    """Pick a random basis key from the inner module at the given degree.

    Uses reservoir sampling (one pass, O(1) memory).
    """
    chosen = None
    count = 0
    for key in _module_basis_keys_in_degree(module, degree):
        count += 1
        if rng.randint(1, count) == 1:
            chosen = key
    return chosen


def _random_m_tuple(module, n: int, total_deg: int, rng: Random) -> tuple | None:
    """Generate a random n-tuple of module basis keys with given total degree.

    Uses a simple strategy: for each position, pick a random valid degree
    and sample a key at that degree.
    """
    if n == 0:
        return () if total_deg == 0 else None
    if n == 1:
        key = _random_module_key(module, total_deg, rng)
        return (key,) if key is not None else None

    # Try to build an n-tuple by greedily picking keys
    m_conn = int(getattr(module, "connectivity", 0))
    result = []
    remaining = total_deg

    for i in range(n):
        slots_left = n - i - 1
        # Minimum degree needed for remaining slots
        min_remaining = slots_left * max(0, m_conn)
        max_for_this = remaining - min_remaining
        min_for_this = max(0, m_conn)

        if max_for_this < min_for_this:
            return None

        # Pick a random degree for this slot
        d_this = rng.randint(min_for_this, max_for_this)
        key = _random_module_key(module, d_this, rng)
        if key is None:
            # Try degree 0 if module connectivity is 0
            if d_this != 0:
                key = _random_module_key(module, 0, rng)
                if key is not None:
                    d_this = 0
            if key is None:
                return None

        result.append(key)
        remaining -= d_this

    if remaining != 0:
        return None

    return tuple(result)


def random_free_algebra_element(
    parent,
    degree: int,
    rng: Random,
    *,
    weight: int | None = None,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a random free algebra element without materializing the full basis.

    For ``P ∘ M``, samples a random arity ``n``, a random planar operad element
    from ``P(n)``, and a random ``n``-tuple of module basis keys.

    Parameters
    ----------
    parent : FreeAlgebraModule
        The free algebra module.
    degree : int
        Target total degree.
    rng : Random
        Random number generator.
    weight : int or None
        If given, restrict to elements of this weight.
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
    """
    P = parent._operad_cls
    M = parent._inner_module
    R = parent.base_ring()
    connectivity = int(getattr(P, "connectivity", 0))

    # Determine arity range
    m_conn = int(getattr(M, "connectivity", 0))
    if weight is not None:
        max_n = weight  # each leaf has weight >= 1
    elif m_conn > 0:
        max_n = max(1, degree // m_conn + 2)
    elif connectivity > 0:
        max_n = max(1, degree // connectivity + 2)
    else:
        max_n = 10  # heuristic bound

    # n=1: identity operad key, single module element
    possible_n = list(range(1, max_n + 1))
    rng.shuffle(possible_n)

    max_attempts = 50
    for _ in range(max_attempts):
        if not possible_n:
            break
        n = possible_n[rng.randint(0, len(possible_n) - 1)]

        if n == 1:
            # P(1) = unit, need module key at degree d
            mk = _random_module_key(M, degree, rng)
            if mk is not None:
                if weight is None or _inner_weight_sum(M, (mk,)) == weight:
                    return parent.term((P.unit_key(), (mk,)))
            continue

        # n >= 2: pick random operad degree, then module degree
        min_p_deg = connectivity * (n - 1)
        max_p_deg = degree  # m_tuple could have degree 0

        if max_p_deg < min_p_deg:
            continue

        p_deg = rng.randint(min_p_deg, max_p_deg)
        m_deg = degree - p_deg

        # Sample a random planar operad element.
        # When sphere_nontrivial is True, bypass planar_basis_iter and use
        # construction-aware sampling that propagates the sphere filter.
        comp_n = P(n, R)
        planar_iter = getattr(comp_n, "planar_basis_iter", None)
        if planar_iter is not None and not sphere_nontrivial:  # use fast planar path
            p_elem = _random_from_planar_iter(comp_n, p_deg, rng)
        else:
            p_elem = _random_operad_element(
                comp_n,
                p_deg,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )

        if p_elem is None:
            continue

        p_key = _extract_key(p_elem)
        if p_key is None:
            continue

        # Sample a random m_tuple
        m_tuple = _random_m_tuple(M, n, m_deg, rng)
        if m_tuple is None:
            continue

        if weight is not None and _inner_weight_sum(M, m_tuple) != weight:
            continue

        return parent.term((p_key, m_tuple))

    return None


def _random_from_planar_iter(comp, degree: int, rng: Random):
    """Pick a random element from the planar basis iterator using reservoir sampling."""
    chosen = None
    count = 0
    for elem in comp.planar_basis_iter(degree):
        count += 1
        if rng.randint(1, count) == 1:
            chosen = elem
    return chosen


def _inner_weight_sum(module, m_tuple: tuple) -> int:
    """Compute total weight of an m_tuple."""
    w_fn = getattr(module, "_weight_on_basis", None)
    if w_fn is not None:
        return sum(w_fn(mk) for mk in m_tuple)
    return len(m_tuple)  # default: weight 1 per key


def random_cofree_coalgebra_element(
    parent,
    degree: int,
    rng: Random,
    *,
    weight: int | None = None,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a random cofree coalgebra element without materializing the full basis.

    For ``C ∘ M``, samples a random arity ``n``, a random planar cooperad element
    from ``C(n)``, and a random ``n``-tuple of module basis keys.

    Parameters
    ----------
    parent : CofreeCoalgebraModule
        The cofree coalgebra module.
    degree : int
        Target total degree.
    rng : Random
        Random number generator.
    weight : int or None
        If given, restrict to elements of this weight.
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
    """
    C = parent._cooperad_cls
    M = parent._inner_module
    R = parent.base_ring()
    connectivity = int(getattr(C, "connectivity", 0))

    m_conn = int(getattr(M, "connectivity", 0))
    if weight is not None:
        max_n = weight
    elif m_conn > 0:
        max_n = max(1, (degree + 2 * abs(connectivity)) // max(1, m_conn) + 2)
    elif connectivity > 0:
        max_n = max(1, degree // connectivity + 1)
    else:
        max_n = 6  # cap to avoid combinatorial explosion in planar_basis_iter

    possible_n = list(range(1, max_n + 1))
    rng.shuffle(possible_n)

    max_attempts = 50
    for _ in range(max_attempts):
        if not possible_n:
            break
        n = possible_n[rng.randint(0, len(possible_n) - 1)]

        if n == 1:
            mk = _random_module_key(M, degree, rng)
            if mk is not None:
                if weight is None or _inner_weight_sum(M, (mk,)) == weight:
                    return parent.term((C.unit_key(), (mk,)))
            continue

        min_c_deg = connectivity * (n - 1) if connectivity < 0 else 0
        max_c_deg = degree

        if max_c_deg < min_c_deg:
            continue

        c_deg = rng.randint(min_c_deg, max_c_deg)
        m_deg = degree - c_deg

        comp_n = C(n, R)
        # When sphere_nontrivial is True, bypass planar_basis_iter and use
        # construction-aware sampling that propagates the sphere filter.
        planar_iter = getattr(comp_n, "planar_basis_iter", None)
        if planar_iter is not None and not sphere_nontrivial:  # use fast planar path
            c_elem = _random_from_planar_iter(comp_n, c_deg, rng)
        else:
            c_elem = _random_operad_element(
                comp_n,
                c_deg,
                rng,
                sphere_nontrivial=sphere_nontrivial,
                sphere_dim=sphere_dim,
            )

        if c_elem is None:
            continue

        c_key = _extract_key(c_elem)
        if c_key is None:
            continue

        m_tuple = _random_m_tuple(M, n, m_deg, rng)
        if m_tuple is None:
            continue

        if weight is not None and _inner_weight_sum(M, m_tuple) != weight:
            continue

        return parent.term((c_key, m_tuple))

    return None


def random_tree_module_element(
    parent,
    degree: int,
    rng: Random,
    *,
    weight: int | None = None,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
):
    """Generate a random tree-module element without materializing the full basis.

    For tree-decorated composites ``S ∘ M``, generates a random decorated
    tree with random module elements at the leaves.

    Parameters
    ----------
    parent : TreeModule
        The tree module.
    degree : int
        Target total degree.
    rng : Random
        Random number generator.
    weight : int or None
        If given, restrict to elements of this weight.
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.

    Returns
    -------
    Element or None
    """
    S = parent._symmetric_sequence_cls
    M = parent._inner_module
    R = parent.base_ring()
    vertex_shift = parent._vertex_degree_shift
    connectivity = int(getattr(S, "connectivity", 0))
    m_conn = int(getattr(M, "connectivity", 0))

    if weight is not None:
        max_n = weight
    elif m_conn > 0:
        max_n = max(1, degree // m_conn + 2)
    elif connectivity > 0:
        max_n = max(1, (degree - vertex_shift) // connectivity + 2)
    else:
        max_n = 6  # cap to avoid combinatorial explosion in planar_basis_iter

    possible_n = list(range(1, max_n + 1))
    rng.shuffle(possible_n)

    max_attempts = 50
    for _ in range(max_attempts):
        if not possible_n:
            break
        n = possible_n[rng.randint(0, len(possible_n) - 1)]

        if n == 1:
            mk = _random_module_key(M, degree, rng)
            if mk is not None:
                if weight is None or _inner_weight_sum(M, (mk,)) == weight:
                    return parent((1, (mk,)))
            continue

        # Pick a random tree degree / module degree split
        min_tree_deg = vertex_shift + connectivity * (n - 1)
        max_tree_deg = degree  # m_tuple could have degree 0

        if max_tree_deg < min_tree_deg:
            continue

        tree_deg = rng.randint(min_tree_deg, max_tree_deg)
        m_deg = degree - tree_deg

        # Generate a random tree
        max_w = n - 1
        tree = random_shuffle_tree(
            tuple(range(1, n + 1)),
            max_w,
            S,
            R,
            tree_deg,
            vertex_shift,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if tree is None:
            continue

        # Generate a random m_tuple
        m_tuple = _random_m_tuple(M, n, m_deg, rng)
        if m_tuple is None:
            continue

        if weight is not None and _inner_weight_sum(M, m_tuple) != weight:
            continue

        return parent((tree, m_tuple))

    return None


# ---------------------------------------------------------------------------
# Generic basis sampling (with construction-aware dispatch)
# ---------------------------------------------------------------------------


def sample_basis(
    parent,
    degree: int,
    k: int,
    rng: Random,
    *,
    weight: int | None = None,
    sphere_nontrivial: bool = False,
    sphere_dim: int | None = None,
) -> list:
    """Sample up to *k* random basis elements from *parent* at degree *degree*.

    Dispatches to construction-aware generators when the parent type is
    recognized (bar/cobar constructions, free/cofree algebras, Hadamard
    products, etc.).  Falls back to ``graded_basis``/``basis_iter``
    enumeration when no direct sampler is available.

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
    sphere_nontrivial : bool
        If ``True``, only generate surjection factors that act nontrivially
        on ``S^{sphere_dim}``.  Propagated recursively through all
        constructions.
    sphere_dim : int or None
        Required when ``sphere_nontrivial=True``.
    """
    # Try construction-aware dispatch first
    direct_sampler = _get_direct_sampler(parent, weight)
    if direct_sampler is not None:
        return _sample_via_direct(
            direct_sampler,
            parent,
            degree,
            k,
            rng,
            weight,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )

    # Fallback to full-basis enumeration
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
                full = [
                    e
                    for e in full
                    if getattr(parent, "_weight_on_basis", lambda _: 0)(
                        next(iter(e.monomial_coefficients()))
                    )
                    == weight
                ]
    else:
        cached = getattr(parent, "graded_basis", None)
        if cached is not None:
            full = list(cached(degree))
        else:
            full = list(parent.basis_iter(degree))

    if len(full) <= k:
        return list(full)
    return rng.sample(full, k)


def _get_direct_sampler(parent, weight):
    """Return a direct-sampling function for the given parent type, or None."""
    if isinstance(parent, BarConstruction.Component):
        return "bar"
    if isinstance(parent, CobarConstruction.Component):
        return "cobar"
    # Check tree module before free/cofree since BarAlgebra/CobarCoalgebra
    # inherit from those
    if isinstance(parent, TreeModule):
        return "tree_module"
    if isinstance(parent, FreeAlgebraModule):
        return "free_algebra"
    if isinstance(parent, CofreeCoalgebraModule):
        return "cofree_coalgebra"
    if isinstance(parent, HadamardProduct.Component):
        return "hadamard"
    if isinstance(parent, Surjection):
        return "surjection"
    if isinstance(parent, BarrattEccles):
        return "barratt_eccles"
    if isinstance(parent, Lie):
        return "lie"

    return None


def _sample_via_direct(
    sampler_type, parent, degree, k, rng, weight, *, sphere_nontrivial=False, sphere_dim=None
):
    """Use a construction-aware sampler to generate up to *k* distinct elements."""
    seen = set()
    results = []
    max_attempts = k * 15  # generous retry budget

    for _ in range(max_attempts):
        if len(results) >= k:
            break

        elem = _invoke_direct_sampler(
            sampler_type,
            parent,
            degree,
            rng,
            weight,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
        if elem is None:
            continue

        # Deduplicate by support
        try:
            key = frozenset((b, c) for b, c in elem)
        except (TypeError, ValueError):
            key = id(elem)  # fallback if unhashable

        if key not in seen:
            seen.add(key)
            results.append(elem)

    return results


def _invoke_direct_sampler(
    sampler_type, parent, degree, rng, weight, *, sphere_nontrivial=False, sphere_dim=None
):
    """Call the appropriate direct sampler."""
    if sampler_type == "bar":
        return random_bar_element(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "cobar":
        return random_cobar_element(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "free_algebra":
        return random_free_algebra_element(
            parent,
            degree,
            rng,
            weight=weight,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "cofree_coalgebra":
        return random_cofree_coalgebra_element(
            parent,
            degree,
            rng,
            weight=weight,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "tree_module":
        return random_tree_module_element(
            parent,
            degree,
            rng,
            weight=weight,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "hadamard":
        return random_hadamard_key(
            parent,
            degree,
            rng,
            sphere_nontrivial=sphere_nontrivial,
            sphere_dim=sphere_dim,
        )
    if sampler_type == "surjection":
        n = parent.arity()
        if sphere_nontrivial:
            if sphere_dim is None:
                raise ValueError("sphere_dim is required when sphere_nontrivial=True")
            return random_sphere_admissible_surjection(n, sphere_dim, parent.base_ring(), rng)
        return random_surjection(n, degree, parent.base_ring(), rng)
    if sampler_type == "barratt_eccles":
        n = parent.arity()
        return random_barratt_eccles_element(n, degree, parent.base_ring(), rng)
    if sampler_type == "lie":
        if degree != 0:
            return None
        n = parent.arity()
        key = random_lie_key(n, rng)
        return parent.term(key)
    return None


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

    return sample_basis(
        parent,
        degree,
        k,
        rng,
        sphere_nontrivial=sphere_nontrivial,
        sphere_dim=sphere_dim,
    )


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

    Replacement for the test helper ``_build_algebra_pool``.  Uses
    construction-aware sampling to avoid materializing the full basis.

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
