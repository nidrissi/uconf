from uconf.algebraic.algebra import OperadAlgebra
from sage.all import CombinatorialFreeModule, GradedModulesWithBasis

from uconf.algebraic.free_algebra import FreeOperadAlgebra
from uconf.algebraic.hadamard_algebra import HadamardTensorAlgebra
from uconf.algebraic.pullback_algebra import PullbackAlgebra
from uconf.algebraic.simplicial import (
    SurjectionSimplicialCochainAlgebra,
)
from uconf.constructions.bar_algebra import BarComplexAlgebra
from uconf.constructions.bar_construction import BarConstruction
from uconf.constructions.cobar_construction import CobarConstruction
from uconf.core.morphism import OperadMorphism
from uconf.models.barratt_eccles import BarrattEccles
from uconf.models.lie import Lie
from uconf.models.surjection import Surjection
from uconf.morphisms.e_comodule_morphism import make_e_comodule_morphism
from uconf.wrappers.hadamard_operad import HadamardProduct
from uconf.wrappers.shifted_operad import ShiftedOperad


def _table_reduction_morphism() -> OperadMorphism:
    """Build the operad morphism ``E → S`` given by table reduction.

    The table reduction maps Barratt–Eccles elements to surjections and is
    a quasi-isomorphism of E∞-operads.  It is monkey-patched onto
    ``BarrattEccles.Element`` in :mod:`uconf.__init__`.
    """
    return OperadMorphism(
        BarrattEccles,
        Surjection,
        lambda elem: elem.table_reduction(),
    )


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
    free_alg = FreeOperadAlgebra(OBXsLie, coefficients, base_ring)

    # The e-comodule morphism Δ: Ω(C) → E ⊙ Ω(C) lands in the Hadamard
    # product with E = BarrattEccles on the left.  To match this target, the
    # tensor algebra must also be over (E ⊙ Ω(C)).  We achieve this by pulling
    # back the Surjection-algebra (manifold_model) along table_reduction to
    # obtain a BarrattEccles-algebra, then forming the Hadamard tensor product
    # with E on the left.
    be_manifold_model = PullbackAlgebra(_table_reduction_morphism(), manifold_model)
    tensor_alg = HadamardTensorAlgebra(be_manifold_model, free_alg)

    comodule_morphism = make_e_comodule_morphism(BXsLie)
    pulled_back = PullbackAlgebra(comodule_morphism, tensor_alg)
    bar = BarComplexAlgebra(pulled_back, base_ring)

    return bar


def unordered_configuration_model(manifold_model: OperadAlgebra, dimension: int):
    assert dimension >= 0, "Dimension must be non-negative."
    R = manifold_model.module.base_ring()
    trivial_module = CombinatorialFreeModule(R, ["*"], category=GradedModulesWithBasis(R))
    trivial_module.degree_on_basis = lambda _: dimension
    trivial_module.boundary = lambda _: trivial_module.zero()
    trivial_module.connectivity = dimension
    return labelled_configuration_model(manifold_model, trivial_module)


def euclidean_unordered_configuration_model(base_ring, dimension: int):
    assert dimension >= 0, "Dimension must be non-negative."
    alg = SurjectionSimplicialCochainAlgebra(dimension, base_ring)
    return unordered_configuration_model(alg, dimension)
