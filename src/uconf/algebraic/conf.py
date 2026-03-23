from uconf.algebraic.algebra import OperadAlgebra
from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.spherical import SurjectionSphereCochainAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.constructions.twisted_complex import TwistedBarComplex
from uconf.core.morphism import OperadMorphism
from uconf.models.lie import Lie
from uconf.models.surjection import Surjection
from uconf.morphisms.canonical_twisting import canonical_projection
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.wrappers.hadamard_operad import HadamardProduct
from uconf.wrappers.shifted_operad import ShiftedOperad


def _make_surjection_comodule_morphism(cooperad_cls) -> OperadMorphism:
    """Build the operad morphism Ω(C) → S ⊙ Ω(C).

    Composes the e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) (where
    E = BarrattEccles) with table reduction on the left factor, yielding
    a morphism into S ⊙ Ω(C) (where S = Surjection).
    """
    be_comodule = make_e_comodule_morphism(cooperad_cls)
    cobar = CobarConstruction(cooperad_cls)
    surj_target = HadamardProduct(Surjection, cobar)

    def _on_element(element):
        # Apply the BE-valued e-comodule morphism
        be_had_elem = be_comodule(element)

        # Apply table_reduction to the left (BarrattEccles) factor
        source_parent = be_had_elem.parent()
        n = source_parent.arity()
        base_ring = source_parent.base_ring()
        target_n = surj_target(n, base_ring)

        result = target_n.zero()
        be_parent = source_parent.left_parent()
        for (be_key, cobar_key), coeff in be_had_elem:
            surj_elem = be_parent.term(be_key).table_reduction()

            for surj_key, surj_coeff in surj_elem:
                result += coeff * surj_coeff * target_n.term((surj_key, cobar_key))

        return result

    return OperadMorphism(cobar, surj_target, _on_element)


def labelled_configuration_model(
    manifold_model: OperadAlgebra, coefficients: CombinatorialFreeModule
):
    assert manifold_model.operad_cls == Surjection, "Manifold model must be a surjection algebra."

    base_ring = coefficients.base_ring()
    assert manifold_model.module.base_ring() == base_ring, (
        "Coefficient module must have the same base ring as the manifold model."
    )

    sLie = ShiftedOperad(Lie, -1)
    XsLie = HadamardProduct(sLie, Surjection)
    BXsLie = BarConstruction(XsLie)
    OBXsLie = CobarConstruction(BXsLie)
    free_alg = FreeOperadAlgebra(OBXsLie, coefficients)

    # The e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) is postcomposed with
    # table reduction on the left factor to obtain Δ': Ω(C) → S ⊙ Ω(C).
    # This way the tensor algebra is built over (S ⊙ Ω(C)) and the
    # manifold_model (a Surjection-algebra) is used directly.
    tensor_alg = HadamardTensorAlgebra(manifold_model, free_alg)

    comodule_morphism = _make_surjection_comodule_morphism(BXsLie)
    pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)
    pi = canonical_projection(pulled_back.operad_cls)
    bar = TwistedBarComplex(pi, pulled_back)

    return bar


class TrivialModule(CombinatorialFreeModule):
    def __init__(self, dimension: int, base_ring):
        super().__init__(base_ring, ["*"], category=GradedModulesWithBasis(base_ring))
        self._dimension = dimension
        self.boundary = lambda _: self.zero()
        self.connectivity = 0
        self.rename(f"K[{dimension}]")

    def degree_on_basis(self, key):
        return self._dimension


def unordered_configuration_model(manifold_model: OperadAlgebra, dimension: int):
    assert dimension >= 0, "Dimension must be non-negative."
    R = manifold_model.module.base_ring()
    return labelled_configuration_model(manifold_model, TrivialModule(dimension, R))


def euclidean_unordered_configuration_model(base_ring, dimension: int):
    assert dimension >= 0, "Dimension must be non-negative."
    alg = SurjectionSphereCochainAlgebra(dimension, base_ring)
    return unordered_configuration_model(alg, dimension)
