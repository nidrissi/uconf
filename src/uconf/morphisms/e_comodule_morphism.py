"""E-comodule morphism Δ: Ω(C) → E ⊗ Ω(C).

For a quasi-planar cooperad C, the E-comodule map defines an operad morphism
from the cobar construction Ω(C) to the Hadamard product E ⊗ Ω(C), where
E = BarrattEccles is the Barratt–Eccles operad.

Since Ω(C) is the free operad T(s⁻¹C̄), the morphism is uniquely determined
by its values on generators (single-vertex trees).  On generators, the map is
computed by :func:`~uconf.constructions.e_comodule.e_comodule_on_generator`.
The extension to arbitrary trees uses the universal property of the free operad.

The cooperad :math:`\\mathcal{C}` must be quasi-planar: its arity-n components
must inherit from :class:`~uconf.core.quasi_planar.QuasiPlanarMixin` and expose
``planarize``, ``boundary``, and ``d_sigma``.  The canonical case is
:class:`~uconf.constructions.bar_construction.BarConstruction` applied to a
Hadamard product :math:`P \\otimes \\mathcal{E}` or :math:`P \\otimes \\Surj`.
"""

from __future__ import annotations

from typing import Any

from sage.all import SymmetricGroup, tensor

from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.cooperad import CooperadComponent, CooperadLike
from uconf.core.morphism import OperadMorphism
from uconf.core.quasi_planar import QuasiPlanarMixin
from uconf.core.trees import children, decoration, is_leaf, vertex_arity
from uconf.models.barratt_eccles import BarrattEccles
from uconf.wrappers.hadamard_operad import HadamardProduct


def e_comodule_on_generator(dec_elem: Any) -> Any:
    """Compute the :math:`\\mathcal{E}_\\nu`-comodule map on a cooperad generator.

    Given an element *dec_elem* :math:`\\in \\mathcal{C}(n)` (planar or not),
    returns

    .. math::

        \\Delta(s^{-1}\\,\\text{dec\\_elem})
          \\in \\mathcal{E}(n) \\otimes \\Omega(\\mathcal{C})(n).

    For planar input the formula is the direct recursive expansion.
    Non-planar input is handled via equivariance:
    :math:`\\Delta(c \\cdot \\sigma) = \\Delta(c) \\cdot \\sigma`,
    where the diagonal :math:`S_n` action acts on both the
    Barratt–Eccles and cobar factors.

    Parameters
    ----------
    dec_elem :
        Element of a quasi-planar cooperad component (planar or not).

    Returns
    -------
    Element of ``tensor([be_component, cobar_component])``.

    Notes
    -----
    Sequences :math:`\\underline\\sigma` that contain the identity permutation
    give :math:`\\rho(\\underline\\sigma) = 0` (degenerate Barratt–Eccles
    element), so they are skipped.  Zero branches of :math:`d_\\sigma` are
    pruned early.  The recursion terminates naturally because each
    :math:`d_\\sigma` reduces the degree by 1 and the cooperad is bounded below.
    """
    cooperad_component: CooperadComponent = dec_elem.parent()
    assert isinstance(cooperad_component, QuasiPlanarMixin), (
        "Expected a quasi-planar cooperad component."
    )

    n = cooperad_component.arity()
    base_ring = cooperad_component.base_ring()
    cobar_component = CobarConstruction(cooperad_component.factory)(n, base_ring)
    be_component = BarrattEccles(n, base_ring)

    S_n = SymmetricGroup(n)
    identity_n = S_n.identity()

    target = tensor([be_component, cobar_component])

    if not dec_elem:
        return target.zero()

    # -----------------------------------------------------------------------
    # Planarize the input: dec_elem = Σ pl_coeff * c_pl ⊗ σ.
    # Compute Δ on each planar component and apply equivariance.
    # -----------------------------------------------------------------------
    planarized = cooperad_component.planarize(dec_elem)

    total_result = target.zero()

    for (planar_key, sigma_key), pl_coeff in planarized:
        sigma = S_n(sigma_key)
        planar_elem = cooperad_component.term(planar_key)

        # Compute Δ on the planar element via the recursive formula.
        planar_result = _delta_on_planar(
            planar_elem, cooperad_component, cobar_component,
            be_component, S_n, identity_n, target,
        )

        # Apply equivariance: Δ(c·σ) = Δ(c)·σ.
        # The diagonal S_n action permutes both BE and cobar factors.
        if sigma == identity_n:
            total_result += pl_coeff * planar_result
        else:
            acted = target.zero()
            for (be_key, cobar_key), t_coeff in planar_result:
                be_perm = be_component.term(be_key).permute(sigma)
                cobar_perm = cobar_component.term(cobar_key).permute(sigma)
                for be_k, be_c in be_perm:
                    for cb_k, cb_c in cobar_perm:
                        acted += t_coeff * be_c * cb_c * target.term((be_k, cb_k))
            total_result += pl_coeff * acted

    return total_result


def _delta_on_planar(
    planar_elem: Any,
    cooperad_component: Any,
    cobar_component: Any,
    be_component: Any,
    S_n: Any,
    identity_n: Any,
    target: Any,
) -> Any:
    """Core recursive computation of Δ on a **planar** cooperad element.

    Parameters
    ----------
    planar_elem :
        A planar element of the cooperad component.
    cooperad_component :
        The arity-n cooperad component.
    cobar_component :
        The arity-n component of the cobar construction Ω(C).
    be_component :
        The arity-n Barratt–Eccles component.
    S_n :
        The symmetric group ``S_n``.
    identity_n :
        The identity element of ``S_n``.
    target :
        The tensor product ``tensor([be_component, cobar_component])``.

    Returns
    -------
    Element of *target*.
    """
    n = cooperad_component.arity()

    result = target.zero()

    def make_cobar_generator(pl_elem, sigma_prod=None):
        """Build the weight-1 cobar tree element for *pl_elem* ∈ C_pl(n).

        If *sigma_prod* is given and is not the identity, the cooperad
        S_n action is applied to the decoration (rather than permuting
        cobar leaves).  This produces ``tree(dec·σ, 1, …, n)`` which
        matches the representation used by the cobar differential d_1.
        """
        acc = cobar_component.zero()
        for dec_key, coeff in pl_elem:
            if sigma_prod is not None and sigma_prod != identity_n:
                acted = cooperad_component(dec_key).permute(sigma_prod)
                for new_key, new_coeff in acted:
                    tree = (new_key,) + tuple(range(1, n + 1))
                    acc += coeff * new_coeff * cobar_component(tree)
            else:
                tree = (dec_key,) + tuple(range(1, n + 1))
                acc += coeff * cobar_component(tree)
        return acc

    def recurse(current_d_elem, sigma_bar):
        """Accumulate contributions, pruning zero branches."""
        nonlocal result

        if not current_d_elem:
            return

        k = len(sigma_bar)

        if k == 0:
            be_elem = be_component((identity_n,))
            cobar_gen = make_cobar_generator(current_d_elem)
            result += be_elem.tensor(cobar_gen)
        else:
            be_elem = be_component.rho(list(sigma_bar))

            if be_elem:
                sigma_prod = identity_n
                for s in sigma_bar:
                    sigma_prod = s * sigma_prod

                cobar_gen = make_cobar_generator(current_d_elem, sigma_prod)
                result += be_elem.tensor(cobar_gen)

        for sigma in S_n:
            if sigma == identity_n:
                continue
            next_d = cooperad_component.d_sigma(current_d_elem, sigma)
            if next_d:
                recurse(next_d, sigma_bar + [sigma])

    recurse(planar_elem, [])
    return result


def make_e_comodule_morphism(
    cooperad_cls: CooperadLike,
) -> OperadMorphism:
    """Build the operad morphism Δ: Ω(C) → E ⊗ Ω(C).

    Parameters
    ----------
    cooperad_cls : CooperadLike
        A quasi-planar cooperad (class or factory) for which the cobar
        construction Ω(C) is defined.

    Returns
    -------
    OperadMorphism
        The morphism Δ whose source is ``CobarConstruction(cooperad_cls)``
        and whose target is ``HadamardProduct(BarrattEccles, CobarConstruction(cooperad_cls))``.
    """
    cobar = CobarConstruction(cooperad_cls)
    target_factory = HadamardProduct(BarrattEccles, cobar)

    def _extend_tree(tree: Any, base_ring: Any) -> Any:
        """Extend the morphism to a single tree by the free operad universal property."""
        if is_leaf(tree):
            return target_factory.unit(base_ring)

        dec = decoration(tree)
        kids = children(tree)
        k = vertex_arity(tree)

        # Map the root generator via e_comodule_on_generator.
        # The decoration may be non-planar; e_comodule_on_generator handles
        # this via internal planarization and equivariance.
        cooperad_parent = cooperad_cls(k, base_ring)
        gen_elem = cooperad_parent(dec)

        root_tensor = e_comodule_on_generator(gen_elem)

        # Convert tensor([BE(k), Ω(C)(k)]) → HadamardProduct element.
        target_k = target_factory(k, base_ring)
        root_image = target_k.zero()
        for tensor_basis, t_coeff in root_tensor:
            be_key, cobar_key = tensor_basis
            root_image += t_coeff * target_k((be_key, cobar_key))

        # Compose with child images from right to left (∘_k, ∘_{k-1}, ..., ∘_1)
        # This preserves input positions 1, ..., j-1 at each step.
        result = root_image
        for j in range(k, 0, -1):
            child = kids[j - 1]
            if is_leaf(child):
                # Composing with the unit is the identity, skip
                continue
            child_image = _extend_tree(child, base_ring)
            result = target_factory.compose(result, j, child_image)

        return result

    def _on_element(element: Any) -> Any:
        """Apply the morphism to an arbitrary cobar element by linearity."""
        parent = element.parent()
        n = parent.arity()
        base_ring = parent.base_ring()
        target_n = target_factory(n, base_ring)
        result = target_n.zero()
        for tree, coeff in element:
            result += coeff * _extend_tree(tree, base_ring)
        return result

    return OperadMorphism(cobar, target_factory, _on_element)
