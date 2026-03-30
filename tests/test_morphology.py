"""Unit tests for morphological operations and utility functions in colabbtr.morphology."""

import numpy as np
import pytest
import torch

from colabbtr.morphology import (
    afmize,
    compute_xc_yc,
    define_tip,
    idilation,
    idilation_old,
    ierosion,
    ierosion_old,
    surfing,
    surfing_old,
    translate_tip_mean,
    Atom2Radius,
    TipShapeMLP,
    SurfaceMLP,
    generate_tip_from_mlp,
    generate_surface_from_mlp,
    BTRLoss,
    SurfaceLoss,
)


# ===== compute_xc_yc =====

class TestComputeXcYc:
    def test_odd_square(self):
        tip = torch.zeros(11, 11)
        xc, yc = compute_xc_yc(tip)
        assert xc == 5
        assert yc == 5

    def test_even_square(self):
        tip = torch.zeros(10, 10)
        xc, yc = compute_xc_yc(tip)
        # (10-1)/2 = 4.5 -> round to 4
        assert xc == 4
        assert yc == 4

    def test_1x1(self):
        tip = torch.zeros(1, 1)
        xc, yc = compute_xc_yc(tip)
        assert xc == 0
        assert yc == 0

    def test_rectangular(self):
        tip = torch.zeros(7, 13)
        xc, yc = compute_xc_yc(tip)
        assert xc == 3
        assert yc == 6


# ===== idilation: old vs new equivalence =====

class TestDilationEquivalence:
    @pytest.fixture
    def tip_and_surface(self):
        """Large surface (64x64) with narrow tip (15x15, R=2.0, 1.0 nm/pixel).
        Tip edge values ≈ -28, well below any surface value."""
        torch.manual_seed(42)
        surface = torch.randn(64, 64, dtype=torch.float64)
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, resolution_x=1.0, resolution_y=1.0, probeRadius=2.0, probeAngle=0.3)
        return surface, tip

    def test_idilation_old_vs_new_interior(self, tip_and_surface):
        """New (-inf pad) and old (periodic) match in the interior, differ at boundary."""
        surface, tip = tip_and_surface
        margin = tip.shape[0] // 2
        result_old = idilation_old(surface, tip)
        result_new = idilation(surface, tip)
        torch.testing.assert_close(
            result_new[margin:-margin, margin:-margin],
            result_old[margin:-margin, margin:-margin],
            atol=1e-10, rtol=1e-10,
        )

    def test_ierosion_old_vs_new_interior(self, tip_and_surface):
        """New (+inf pad) and old (periodic) match in the interior."""
        surface, tip = tip_and_surface
        margin = tip.shape[0] // 2
        image = idilation(surface, tip)
        result_old = ierosion_old(image, tip)
        result_new = ierosion(image, tip)
        torch.testing.assert_close(
            result_new[margin:-margin, margin:-margin],
            result_old[margin:-margin, margin:-margin],
            atol=1e-10, rtol=1e-10,
        )

    def test_dilation_boundary_no_wraparound(self):
        """New version should NOT wrap values from opposite edge (unlike old periodic version)."""
        surface = torch.zeros(64, 64, dtype=torch.float64)
        surface[63, 63] = 100.0  # large value in bottom-right corner
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        result_new = idilation(surface, tip)
        result_old = idilation_old(surface, tip)
        # Old (periodic) wraps the large value to the top-left corner
        assert result_old[0, 0].item() > 50.0
        # New (-inf pad) does NOT wrap — top-left stays at 0
        assert result_new[0, 0].item() == pytest.approx(0.0)

    def test_small_tip(self):
        """Test with 3x3 minimum tip size — interior must match."""
        torch.manual_seed(0)
        surface = torch.randn(64, 64, dtype=torch.float64)
        tip = torch.tensor([[-1., -1., -1.], [-1., 0., -1.], [-1., -1., -1.]], dtype=torch.float64)
        result_old = idilation_old(surface, tip)
        result_new = idilation(surface, tip)
        margin = 1
        torch.testing.assert_close(
            result_new[margin:-margin, margin:-margin],
            result_old[margin:-margin, margin:-margin],
            atol=1e-10, rtol=1e-10,
        )


# ===== surfing: old vs new equivalence =====

class TestSurfingEquivalence:
    def test_surfing_old_vs_new(self):
        """Large grid (±10 nm, 1.0 nm/pixel) with molecules well inside."""
        torch.manual_seed(0)
        N = 10
        xyz = torch.randn(N, 3, dtype=torch.float32)
        xyz[:, 2] = xyz[:, 2].abs()  # positive z
        radii = torch.full((N,), 0.5, dtype=torch.float32)
        config = {
            "min_x": -10.0, "max_x": 10.0,
            "min_y": -10.0, "max_y": 10.0,
            "resolution_x": 1.0, "resolution_y": 1.0,
        }
        result_old = surfing_old(xyz, radii, config)
        result_new = surfing(xyz, radii, config)
        torch.testing.assert_close(result_new, result_old, atol=1e-5, rtol=1e-5)


# ===== translate_tip_mean =====

class TestTranslateTipMean:
    def test_already_centered(self):
        """Symmetric narrow tip should stay roughly the same."""
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        result = translate_tip_mean(tip)
        torch.testing.assert_close(result, tip, atol=1e-10, rtol=1e-10)

    def test_all_equal(self):
        """Constant tip: edge case where weight is all zero after subtracting min."""
        tip = torch.full((15, 15), -5.0, dtype=torch.float64)
        result = translate_tip_mean(tip)
        assert result.shape == tip.shape

    def test_shifted_tip(self):
        """Off-center tip should be re-centered."""
        tip = torch.full((15, 15), -30.0, dtype=torch.float64)
        tip[1, 1] = 0.0  # peak in top-left
        result = translate_tip_mean(tip)
        xc, yc = compute_xc_yc(tip)
        assert result[xc, yc] > result[0, 0]


# ===== Atom2Radius consistency =====

class TestAtom2Radius:
    def test_carbon_radii_consistent(self):
        """All carbon atom radii should be 0.170 nm."""
        carbon_atoms = [k for k in Atom2Radius if k.startswith("C") and k not in ("CL", "CYS")]
        for atom in carbon_atoms:
            assert Atom2Radius[atom] == pytest.approx(0.170), \
                f"{atom} has radius {Atom2Radius[atom]}, expected 0.170"

    def test_nitrogen_radii_consistent(self):
        """All nitrogen atom radii should be 0.155 nm."""
        nitrogen_atoms = [k for k in Atom2Radius if k.startswith("N")]
        for atom in nitrogen_atoms:
            assert Atom2Radius[atom] == pytest.approx(0.155), \
                f"{atom} has radius {Atom2Radius[atom]}, expected 0.155"

    def test_oxygen_radii_consistent(self):
        """All oxygen atom radii should be 0.152 nm."""
        oxygen_atoms = [k for k in Atom2Radius if k.startswith("O")]
        for atom in oxygen_atoms:
            assert Atom2Radius[atom] == pytest.approx(0.152), \
                f"{atom} has radius {Atom2Radius[atom]}, expected 0.152"


# ===== define_tip =====

class TestDefineTip:
    def test_max_at_center(self):
        """Tip maximum should be 0 (at center after normalization)."""
        tip = torch.zeros(15, 15)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        assert tip.max().item() == pytest.approx(0.0)

    def test_symmetric(self):
        """Tip should be symmetric."""
        tip = torch.zeros(15, 15)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        torch.testing.assert_close(tip, tip.flip(0), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(tip, tip.flip(1), atol=1e-10, rtol=1e-10)

    def test_non_positive(self):
        """All tip values should be <= 0."""
        tip = torch.zeros(15, 15)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        assert (tip <= 0).all()

    def test_edges_deeply_negative(self):
        """Tip boundary values should be deeply negative (narrow tip)."""
        tip = torch.zeros(15, 15)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        assert tip[0, 0].item() < -20.0


# ===== afmize =====

class TestAfmize:
    def test_output_shape(self):
        xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        radii = torch.tensor([0.5], dtype=torch.float32)
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        config = {
            "min_x": -10.0, "max_x": 10.0,
            "min_y": -10.0, "max_y": 10.0,
            "resolution_x": 1.0, "resolution_y": 1.0,
        }
        image = afmize(xyz, tip, radii, config)
        expected_w = len(torch.arange(-10.0, 10.0, 1.0))
        expected_h = len(torch.arange(-10.0, 10.0, 1.0))
        assert image.shape == (expected_h, expected_w)


# ===== PINN models =====

class TestTipShapeMLP:
    def test_forward_shape(self):
        model = TipShapeMLP(n_size=10, n_hidden_layers=2, n_nodes=32)
        x = torch.randn(100)
        y = torch.randn(100)
        t = torch.zeros(100)
        out = model(x, y, t)
        assert out.shape == (100, 1)

    def test_generate_tip_shape(self):
        model = TipShapeMLP(n_size=10, n_hidden_layers=2, n_nodes=32)
        model.eval()
        xc = torch.tensor(0.0)
        yc = torch.tensor(0.0)
        t = torch.zeros(100)
        tip = generate_tip_from_mlp(model, kernel_size=10, t=t, xc=xc, yc=yc, device=None)
        assert tip.shape == (10, 10)

    def test_gradient_flows(self):
        model = TipShapeMLP(n_size=10, n_hidden_layers=2, n_nodes=32)
        model.train()
        x = torch.randn(25)
        y = torch.randn(25)
        t = torch.zeros(25)
        out = model(x, y, t)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestSurfaceMLP:
    def test_forward_shape(self):
        model = SurfaceMLP(n_hidden_layers=2, n_nodes=32)
        x = torch.randn(50)
        y = torch.randn(50)
        t = torch.zeros(50)
        out = model(x, y, t)
        assert out.shape == (50, 1)

    def test_generate_surface(self):
        model = SurfaceMLP(n_hidden_layers=2, n_nodes=32)
        model.eval()
        x = torch.linspace(-5, 5, 10)
        y = torch.linspace(-5, 5, 10)
        t = torch.zeros(10, 10)
        surface = generate_surface_from_mlp(model, x, y, t, device=None)
        assert surface.shape == (10, 10)


# ===== Mathematical properties =====

class TestMathematicalProperties:
    def test_dilation_erosion_roundtrip(self):
        """Opening of a dilated image should recover it: dilation(erosion(dilation(s,t),t),t) == dilation(s,t)."""
        torch.manual_seed(42)
        surface = torch.randn(64, 64, dtype=torch.float64).abs()
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        image = idilation(surface, tip)
        image2 = idilation(ierosion(image, tip), tip)
        torch.testing.assert_close(image, image2, atol=1e-10, rtol=1e-10)

    def test_erosion_reduces_values(self):
        """Erosion should generally reduce or maintain height values."""
        torch.manual_seed(42)
        image = torch.randn(64, 64, dtype=torch.float64).abs() + 1.0
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        eroded = ierosion(image, tip)
        assert eroded.max() <= image.max() + 1e-10

    def test_dilation_increases_values(self):
        """Dilation should generally increase or maintain height values."""
        torch.manual_seed(42)
        surface = torch.randn(64, 64, dtype=torch.float64)
        tip = torch.zeros(15, 15, dtype=torch.float64)
        tip = define_tip(tip, 1.0, 1.0, 2.0, 0.3)
        dilated = idilation(surface, tip)
        assert (dilated >= surface - 1e-10).all()
