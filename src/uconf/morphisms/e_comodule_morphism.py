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
    """Compute the :math:`\\mathcal{E}_\\nu`-comodule map on a planar generator.

    Given a **planar** element *dec_elem* :math:`\\in \\mathcal{C}_\\mathrm{pl}(n)`,
    returns

    .. math::

        \\Delta(s^{-1}\\,\\text{dec\\_elem})
          = \\sum_{k \\ge 0}\\;
            \\sum_{\\underline\\sigma \\in (S_n \\setminus \\{\\mathrm{id}\\})^k}
            \\rho(\\underline\\sigma) \\otimes
            \\operatorname{cobar}(d_{\\underline\\sigma}(\\text{dec\\_elem}))
            \\cdot \\sigma_1 \\cdots \\sigma_k

    as an element of :math:`\\mathcal{E}(n) \\otimes \\Omega(\\mathcal{C})(n)`.

    Parameters
    ----------
    dec_elem :
        **Planar** element of *cooperad_component*.

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

    result = target.zero()

    # -----------------------------------------------------------------------
    # Local helpers
    # -----------------------------------------------------------------------

    def make_cobar_generator(planar_elem, sigma_prod=None):
        """Build the weight-1 cobar tree element for *planar_elem* ∈ C(n).

        If *sigma_prod* is given and is not the identity, the cooperad
        S_n action is applied to the decoration (rather than permuting
        cobar leaves).  This produces ``tree(dec·σ, 1, …, n)`` which
        matches the representation used by the cobar differential d_1.
        """
        acc = cobar_component.zero()
        for dec_key, coeff in planar_elem:
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
        """Accumulate contributions, pruning zero branches.

        Parameters
        ----------
        current_d_elem :
            Element :math:`d_{\\underline\\sigma}(\\text{dec\\_elem})` computed
            so far (a planar element of *cooperad_component*).
        sigma_bar :
            The list :math:`[\\sigma_1, \\ldots, \\sigma_k]` accumulated so far.

        """
        nonlocal result

        if not current_d_elem:
            return

        k = len(sigma_bar)

        # --- k = 0 term: ρ(()) = (id,), no permutation --------------------
        if k == 0:
            be_elem = be_component((identity_n,))
            cobar_gen = make_cobar_generator(current_d_elem)
            result += be_elem.tensor(cobar_gen)
        else:
            # ρ(σ̄) = BE_n.rho([σ_1, σ_2, ..., σ_k])
            be_elem = be_component.rho(list(sigma_bar))

            if be_elem:
                # σ_prod = σ_1 · σ_2 · ... · σ_k
                sigma_prod = sigma_bar[0]
                for s in sigma_bar[1:]:
                    sigma_prod = sigma_prod * s

                # Act σ_prod on the cooperad decoration rather than
                # permuting cobar leaf labels.  This ensures the output
                # uses the same representation as the cobar d_1.
                cobar_gen = make_cobar_generator(current_d_elem, sigma_prod)
                result += be_elem.tensor(cobar_gen)

        # --- Recurse: extend sigma_bar by one more non-identity perm -------
        # Terminates naturally: each d_sigma reduces degree by 1, and the
        # cooperad is bounded below at each fixed arity, so eventually
        # all d_sigma return zero and recursion stops.
        for sigma in S_n:
            if sigma == identity_n:
                continue  # identity always makes ρ = 0
            next_d = cooperad_component.d_sigma(current_d_elem, sigma)
            if next_d:  # prune zero branches
                recurse(next_d, sigma_bar + [sigma])

    recurse(dec_elem, [])
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
        # The decoration may be non-planar (e.g. when the cobar differential
        # produces trees with non-planar vertex decorations).  We planarize
        # first, apply the formula to each planar component, then use
        # equivariance: Δ(s⁻¹(c·σ)) = Δ(s⁻¹c) · σ.
        cooperad_parent = cooperad_cls(k, base_ring)
        gen_elem = cooperad_parent(dec)
        planarized = cooperad_parent.planarize(gen_elem)

        S_k = SymmetricGroup(k)
        identity_k = S_k.identity()

        target_k = target_factory(k, base_ring)
        root_image = target_k.zero()

        be_component_k = BarrattEccles(k, base_ring)

        for (planar_key, sigma_key), pl_coeff in planarized:
            sigma = S_k(sigma_key)
            planar_elem = cooperad_parent(planar_key)

            root_tensor = e_comodule_on_generator(planar_elem)

            # Convert tensor([BE(k), Ω(C)(k)]) → HadamardProduct element,
            # applying equivariance by sigma.
            #
            # Equivariance: Δ(s⁻¹(c·σ)) = Δ(s⁻¹c) · σ.
            # The diagonal S_k action on the tensor product E⊗Ω(C) acts
            # on BE by .permute(σ) and on the cobar factor by acting σ
            # on the cooperad decoration (NOT by permuting leaf labels).
            # This ensures that the cobar factor always lives in the
            # same representation as the output of the cobar differential d_1,
            # which uses non-planar decorations with standard leaf order.
            partial = target_k.zero()
            for tensor_basis, t_coeff in root_tensor:
                be_key, cobar_key = tensor_basis

                if sigma != identity_k:
                    # BE factor: standard permutation
                    be_perm = be_component_k(be_key).permute(sigma)

                    # Cobar factor: act σ on the cooperad decoration.
                    # cobar_key is a weight-1 tree (dec, 1, …, k).
                    inner_dec = cobar_key[0]
                    acted_dec = cooperad_parent(inner_dec).permute(sigma)

                    for be_k, be_c in be_perm:
                        for new_dec, dec_c in acted_dec:
                            new_cobar_key = (new_dec,) + cobar_key[1:]
                            partial += t_coeff * be_c * dec_c * target_k((be_k, new_cobar_key))
                else:
                    partial += t_coeff * target_k((be_key, cobar_key))

            root_image += pl_coeff * partial

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
