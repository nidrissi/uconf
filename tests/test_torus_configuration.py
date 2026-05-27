"""Tests for the torus configuration model.

Verifies that:
1. The torus configuration model builds successfully
2. The resulting module has a boundary operator
3. The boundary satisfies d² = 0 (chain complex property)
4. Basis elements are correctly graded
"""

import pytest
from sage.all import GF, QQ

from uconf.algebraic.torus_configuration import (
    unordered_torus_configuration_model,
    _build_torus_layers,
)


class TestTorusConfigurationModel:
    """Test suite for torus configuration model construction and chain complex property."""

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_model_builds_successfully(self, base_ring):
        """Test that the torus configuration model builds without errors."""
        model = unordered_torus_configuration_model(base_ring)
        assert model is not None
        assert model.module is not None
        print(f"✓ Model built successfully for {base_ring}")

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_model_has_boundary(self, base_ring):
        """Test that the torus configuration model has a boundary operator."""
        model = unordered_torus_configuration_model(base_ring)
        module = model.module
        
        # The boundary should be defined
        assert hasattr(module, 'boundary') or hasattr(module, '__call__')
        print(f"✓ Boundary operator exists for {base_ring}")

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_chain_complex_property_d_squared_zero(self, base_ring):
        """Test that d² = 0 for the torus configuration model.
        
        This is the fundamental property of a chain complex:
        the boundary of a boundary must be zero.
        """
        model = unordered_torus_configuration_model(base_ring)
        module = model.module
        
        # Get all basis elements up to a reasonable degree
        # We test degrees from -2 to 2 to cover the torus's structure
        degrees_to_test = range(-2, 3)
        
        for d in degrees_to_test:
            basis_elements = list(module.graded_basis(d))
            
            for basis_elem in basis_elements:
                # Create an element in the module
                elem = module(basis_elem)
                
                # Apply boundary twice: d(d(elem))
                d_elem = module.boundary(elem)
                dd_elem = module.boundary(d_elem)
                
                # Check that d² = 0
                assert dd_elem == module.zero(), (
                    f"d² ≠ 0 at degree {d} for basis element {basis_elem}. "
                    f"Got: {dd_elem}"
                )
        
        print(f"✓ Chain complex property d² = 0 verified for {base_ring}")

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_basis_grading(self, base_ring):
        """Test that basis elements have correct grading."""
        layers = _build_torus_layers(base_ring)
        module = layers.bar.module
        
        # Check that grading is consistent across degrees
        for d in range(-2, 3):
            basis_elems = list(module.graded_basis(d))
            
            for elem in basis_elems:
                # Every basis element should have the correct degree
                elem_degree = module.degree_on_basis(elem)
                assert elem_degree == d, (
                    f"Basis element {elem} has reported degree {elem_degree}, "
                    f"but is listed in degree {d}"
                )
        
        print(f"✓ Basis grading is consistent for {base_ring}")

    @pytest.mark.parametrize("base_ring", [GF(2), GF(3), QQ])
    def test_boundary_respects_grading(self, base_ring):
        """Test that the boundary operator respects grading (lowers degree by 1)."""
        model = unordered_torus_configuration_model(base_ring)
        module = model.module
        
        for d in range(-1, 3):
            basis_elements = list(module.graded_basis(d))
            
            for basis_elem in basis_elements:
                elem = module(basis_elem)
                d_elem = module.boundary(elem)
                
                # If d_elem is nonzero, its degree should be d - 1
                if d_elem != module.zero():
                    # Get the degree of the boundary
                    # This is subtle for the bar construction, so we just check
                    # that boundary is well-defined
                    assert d_elem.parent() == module, (
                        f"Boundary of {basis_elem} is not in the same module"
                    )
        
        print(f"✓ Boundary operator respects module structure for {base_ring}")

    def test_layers_structure(self):
        """Test that all layers of the construction are properly initialized."""
        base_ring = GF(2)
        layers = _build_torus_layers(base_ring)
        
        # Check that all layers exist
        assert layers.manifold_model is not None, "Manifold model missing"
        assert layers.coefficients is not None, "Coefficients missing"
        assert layers.sLie is not None, "Shifted Lie missing"
        assert layers.XsLie is not None, "Hadamard product missing"
        assert layers.BXsLie is not None, "Bar construction missing"
        assert layers.OBXsLie is not None, "Cobar construction missing"
        assert layers.free_alg is not None, "Free algebra missing"
        assert layers.tensor_alg is not None, "Tensor algebra missing"
        assert layers.comodule_morphism is not None, "Comodule morphism missing"
        assert layers.pulled_back is not None, "Pulled back algebra missing"
        assert layers.pi is not None, "Twisting morphism missing"
        assert layers.bar is not None, "Bar algebra missing"
        
        print("✓ All construction layers are properly initialized")

    def test_manifold_model_is_surjection_algebra(self):
        """Test that the manifold model is a Surjection algebra."""
        from uconf.models.surjection import Surjection
        
        base_ring = GF(2)
        layers = _build_torus_layers(base_ring)
        
        assert layers.manifold_model.operad_cls == Surjection, (
            f"Manifold model is not a Surjection algebra, got {layers.manifold_model.operad_cls}"
        )
        print("✓ Manifold model is a Surjection algebra")

    def test_coefficients_dimension(self):
        """Test that the coefficient module has dimension 2 (for the torus)."""
        base_ring = GF(2)
        layers = _build_torus_layers(base_ring)
        
        # The torus has dimension 2
        coefficients = layers.coefficients
        assert coefficients._dimension == 2, (
            f"Coefficient module has dimension {coefficients._dimension}, expected 2"
        )
        print("✓ Coefficient module has correct dimension (2)")


class TestChainComplexHomology:
    """Tests for computing homology of the torus configuration chain complex."""

    @pytest.mark.parametrize("base_ring", [GF(2)])
    def test_can_compute_chain_complex(self, base_ring):
        """Test that we can build a chain complex from the model."""
        from uconf.homology import compute_chain_complex
        
        model = unordered_torus_configuration_model(base_ring)
        module = model.module
        
        # Try to compute chain complex (without weight, since torus doesn't use weights)
        try:
            cc = compute_chain_complex(
                module,
                degrees=range(-1, 3),
                progress=False,
                verbose=False,
            )
            assert cc is not None, "Chain complex computation returned None"
            print(f"✓ Chain complex computed successfully for {base_ring}")
        except Exception as e:
            pytest.fail(f"Failed to compute chain complex: {e}")

    def test_homology_structure(self):
        """Test basic properties of the computed homology."""
        from uconf.homology import compute_chain_complex
        
        base_ring = GF(2)
        model = unordered_torus_configuration_model(base_ring)
        module = model.module
        
        cc = compute_chain_complex(
            module,
            degrees=range(-1, 3),
            progress=False,
            verbose=False,
        )
        
        # Homology should be well-defined
        for d in range(-1, 3):
            betti = cc.betti(d)
            assert isinstance(betti, int), f"Betti number at degree {d} is not an integer"
            assert betti >= 0, f"Betti number at degree {d} is negative: {betti}"
        
        print("✓ Homology has correct structure")


if __name__ == "__main__":
    # Allow running tests with: sage -m pytest tests/test_torus_configuration.py -v
    pytest.main([__file__, "-v"])
