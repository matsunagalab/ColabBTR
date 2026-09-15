"""Unit tests for morphological operations and utility functions in colabbtr.morphology."""

import numpy as np
import pytest
import torch

from colabbtr.morphology import (
    afmize,
    afmize_supersampled,
    compute_xc_yc,
    crop_tip,
    define_tip,
    fine_config,
    idilation,
    idilation_old,
    ierosion,
    ierosion_old,
    surfing,
    surfing_old,
    surfing_supersampled,
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
        """Large grid (±10 nm, 1.0 nm/pixel) with molecules well inside.

        Equivalence of the two implementations only — both sample at pixel
        centers, so this says nothing about physical accuracy. See
        TestSurfingSupersampled for that.
        """
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

CONFIG_10 = {
    "min_x": -10.0, "max_x": 10.0,
    "min_y": -10.0, "max_y": 10.0,
    "resolution_x": 1.0, "resolution_y": 1.0,
}


def _tip15(dtype=torch.float64):
    return define_tip(torch.zeros(15, 15, dtype=dtype), 1.0, 1.0, 2.0, 0.3)


class TestAfmize:
    def test_output_shape(self):
        # bead sitting on a pixel center (centers are at +/-0.5, +/-1.5, ...),
        # so the image is actually non-trivial rather than uniformly zero
        xyz = torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float32)
        radii = torch.tensor([0.5], dtype=torch.float32)
        config = CONFIG_10
        image = afmize(xyz, _tip15(), radii, config)
        expected_w = len(torch.arange(-10.0, 10.0, 1.0))
        expected_h = len(torch.arange(-10.0, 10.0, 1.0))
        assert image.shape == (expected_h, expected_w)
        assert (image > 0).any(), "test molecule must actually render"

    def test_bead_between_pixel_centers_disappears(self):
        """Documents the limitation afmize_supersampled exists to fix.

        A bead of radius 0.5 nm centered at the origin on a 1 nm grid has no
        pixel center inside it (the nearest are at distance sqrt(0.5) = 0.707),
        so point sampling renders literally nothing.
        """
        xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        radii = torch.tensor([0.5], dtype=torch.float32)
        assert not (afmize(xyz, _tip15(), radii, CONFIG_10) > 0).any()


# ===== supersampled AFM simulation =====

class TestFineConfig:
    def test_divides_pitch(self):
        fine = fine_config(CONFIG_10, 5)
        assert fine["resolution_x"] == pytest.approx(0.2)
        assert fine["resolution_y"] == pytest.approx(0.2)
        assert fine["min_x"] == CONFIG_10["min_x"]  # domain unchanged

    @pytest.mark.parametrize("factor", [0, 2, 4, -1])
    def test_rejects_non_odd_factor(self, factor):
        with pytest.raises(ValueError):
            fine_config(CONFIG_10, factor)


class TestCropTip:
    def test_dilation_unchanged(self):
        """Cropped elements can never win the max, so the result is identical."""
        torch.manual_seed(0)
        surface = torch.rand(20, 20, dtype=torch.float64) * 3.0
        tip = define_tip(torch.zeros(31, 31, dtype=torch.float64), 0.2, 0.2, 2.0, 0.3)
        cropped = crop_tip(tip, float(surface.max()))
        assert cropped.shape[0] < tip.shape[0], "nothing was cropped"
        assert cropped.shape[0] % 2 == 1
        torch.testing.assert_close(idilation(surface, cropped),
                                   idilation(surface, tip), atol=0.0, rtol=0.0)

    def test_keeps_apex(self):
        tip = define_tip(torch.zeros(31, 31, dtype=torch.float64), 0.2, 0.2, 2.0, 0.3)
        cropped = crop_tip(tip, 1.0)
        xc, yc = compute_xc_yc(cropped)
        assert cropped[xc, yc].item() == pytest.approx(0.0)

    def test_tall_surface_crops_nothing(self):
        tip = define_tip(torch.zeros(15, 15, dtype=torch.float64), 1.0, 1.0, 2.0, 0.3)
        assert crop_tip(tip, 1e6).shape == tip.shape

    def test_negative_surface_needs_actual_minimum(self):
        """A surface that dips below 0 (shift_z=False MD data) widens the bound.

        An isolated spike over a deep plain is the case that separates them: far
        from the spike the apex only reaches -2, so a distant tip element that
        reaches the spike still wins. Assuming min(surface) == 0 discards it.
        """
        surface = torch.full((21, 21), -2.0, dtype=torch.float64)
        surface[10, 10] = 3.0
        tip = define_tip(torch.zeros(21, 21, dtype=torch.float64), 1.0, 1.0, 1.0, 0.3)
        full = idilation(surface, tip)

        correct = crop_tip(tip, float(surface.max()), float(surface.min()))
        torch.testing.assert_close(idilation(surface, correct), full,
                                   atol=0.0, rtol=0.0)

        too_tight = crop_tip(tip, float(surface.max()))  # min defaults to 0
        assert too_tight.shape[0] < correct.shape[0]
        assert not torch.equal(idilation(surface, too_tight), full)

    def test_arbitrary_tip_apex_not_zero(self):
        """A tip not normalised to apex 0 is fine — the bound is relative."""
        torch.manual_seed(6)
        surface = torch.rand(20, 20, dtype=torch.float64) * 3.0
        tip = define_tip(torch.zeros(31, 31, dtype=torch.float64), 0.2, 0.2, 2.0, 0.3)
        shifted = tip + 5.0  # apex at +5 instead of 0
        cropped = crop_tip(shifted, float(surface.max()))
        assert cropped.shape[0] < shifted.shape[0]
        torch.testing.assert_close(idilation(surface, cropped),
                                   idilation(surface, shifted), atol=0.0, rtol=0.0)

    def test_off_centre_maximum(self):
        """The bound must be anchored on tip(centre), not max(tip).

        Only the centre offset is in range at every pixel — idilation pads with
        -inf — so when the highest point of the tip sits elsewhere, anchoring on
        max(tip) drops offsets that still win near the edges.
        """
        torch.manual_seed(0)
        tip = torch.full((5, 5), -100.0, dtype=torch.float64)
        tip[1, 2] = 0.0  # maximum one cell off centre
        surface = torch.rand(12, 12, dtype=torch.float64) * 3.0
        cropped = crop_tip(tip, float(surface.max()), float(surface.min()))
        torch.testing.assert_close(idilation(surface, cropped),
                                   idilation(surface, tip), atol=0.0, rtol=0.0)

    def test_degenerate_shapes(self):
        tip = torch.zeros(1, 1, dtype=torch.float64)
        assert crop_tip(tip, 5.0).shape == (1, 1)
        # idilation needs a square kernel; a non-square tip must at least not
        # be cropped into something with different offsets
        assert crop_tip(torch.rand(3, 5, dtype=torch.float64) - 1, 0.5).shape == (3, 5)

    def test_rejects_inverted_height_range(self):
        with pytest.raises(ValueError, match="exceeds max_surface_height"):
            crop_tip(torch.zeros(5, 5, dtype=torch.float64), 1.0, 5.0)

    def test_fuzz_matches_full_dilation(self):
        """Random tips and surfaces, including even sizes and negative surfaces.

        Even sizes matter because compute_xc_yc rounds (n-1)/2 while idilation
        pads with (n-1)//2; disagreeing about the centre shifts every offset.
        """
        torch.manual_seed(11)
        cropped_any = 0
        for trial in range(400):
            n = int(torch.randint(1, 9, (1,)))
            m = int(torch.randint(1, 9, (1,)))
            tip = torch.rand(n, n, dtype=torch.float64) * 40 - 40
            if trial % 3 == 0:  # sometimes put the maximum somewhere arbitrary
                tip[torch.randint(0, n, (1,)), torch.randint(0, n, (1,))] = 0.0
            surface = torch.rand(m, m, dtype=torch.float64) * 5.0
            if trial % 2 == 0:
                surface = surface - torch.rand(1).double() * 4  # dips below zero
            cropped = crop_tip(tip, float(surface.max()), float(surface.min()))
            cropped_any += cropped.shape[0] < n
            assert torch.equal(idilation(surface, cropped),
                               idilation(surface, tip)), \
                f"trial {trial}: tip {tuple(tip.shape)} -> {tuple(cropped.shape)}"
        assert cropped_any > 0, "fuzz never actually cropped anything"


class TestSurfingSupersampled:
    @pytest.mark.parametrize("factor", [3, 5, 9])
    def test_fine_centers_coincide_with_coarse_centers(self, factor):
        """The invariant that makes offset::factor sampling valid.

        For odd factor every coarse pixel center is also a fine pixel center, so
        reading the fine grid at offset factor//2 must reproduce the coarse grid
        — including through the y-axis flip surfing applies at the end. Agreement
        is to float32 rounding rather than bit-exact: torch.arange accumulates a
        step like 1/3 that is not representable, putting the fine centers ~5e-7 nm
        off the coarse ones. A real misalignment would be off by a whole pixel.
        """
        torch.manual_seed(7)
        xyz = torch.randn(15, 3) * 2.0
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((15,), 0.6)  # big enough that pixel centers do hit
        coarse = surfing(xyz, radii, CONFIG_10)
        fine = surfing(xyz, radii, fine_config(CONFIG_10, factor))
        off = factor // 2
        torch.testing.assert_close(fine[off::factor, off::factor], coarse,
                                   atol=1e-5, rtol=0.0)
        assert (coarse > 0).any(), "test molecule must be visible on the coarse grid"

    def test_factor_1_matches_surfing(self):
        torch.manual_seed(0)
        xyz = torch.randn(10, 3)
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((10,), 0.3)
        torch.testing.assert_close(surfing_supersampled(xyz, radii, CONFIG_10, 1),
                                   surfing(xyz, radii, CONFIG_10),
                                   atol=0.0, rtol=0.0)

    def test_never_below_point_sampled(self):
        """Per-pixel max can only add height relative to the center sample."""
        torch.manual_seed(1)
        xyz = torch.randn(20, 3)
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((20,), 0.29)  # mean CA bead radius, < pitch/2
        coarse = surfing(xyz, radii, CONFIG_10)
        fine = surfing_supersampled(xyz, radii, CONFIG_10, 9)
        assert fine.shape == coarse.shape
        assert (fine >= coarse - 1e-6).all()
        assert (fine > coarse + 1e-3).any(), "supersampling should change something"

    def test_recovers_bead_missed_by_pixel_centers(self):
        xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        radii = torch.tensor([0.5], dtype=torch.float32)
        assert surfing(xyz, radii, CONFIG_10).max().item() == 0.0
        # shift_z puts the bead center at z=0, so the true apex is the radius
        assert surfing_supersampled(xyz, radii, CONFIG_10, 9).max().item() == \
            pytest.approx(0.5, abs=0.01)

    def test_batched(self):
        torch.manual_seed(2)
        xyz = torch.randn(3, 10, 3)
        xyz[..., 2] = xyz[..., 2].abs()
        radii = torch.full((10,), 0.3)
        out = surfing_supersampled(xyz, radii, CONFIG_10, 5)
        assert out.shape == (3,) + tuple(surfing(xyz[0], radii, CONFIG_10).shape)


class TestAfmizeSupersampled:
    def test_factor_1_matches_afmize(self):
        torch.manual_seed(0)
        xyz = torch.randn(10, 3)
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((10,), 0.4)
        image, surface = afmize_supersampled(xyz, radii, CONFIG_10, 2.0, 0.3, 15,
                                             factor=1, crop=False)
        torch.testing.assert_close(image, afmize(xyz, _tip15(), radii, CONFIG_10),
                                   atol=0.0, rtol=0.0)
        torch.testing.assert_close(surface, surfing(xyz, radii, CONFIG_10),
                                   atol=0.0, rtol=0.0)

    def test_crop_does_not_change_result(self):
        torch.manual_seed(0)
        xyz = torch.randn(10, 3)
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((10,), 0.3)
        args = (xyz, radii, CONFIG_10, 2.0, 0.3, 15)
        torch.testing.assert_close(afmize_supersampled(*args, factor=5, crop=True)[0],
                                   afmize_supersampled(*args, factor=5, crop=False)[0],
                                   atol=0.0, rtol=0.0)

    def test_renders_bead_that_afmize_misses(self):
        xyz = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        radii = torch.tensor([0.5], dtype=torch.float32)
        assert not (afmize(xyz, _tip15(), radii, CONFIG_10) > 0).any()
        image, _ = afmize_supersampled(xyz, radii, CONFIG_10, 2.0, 0.3, 15, factor=9)
        assert (image > 0).any()

    def test_image_is_at_least_the_coarse_rendering(self):
        """A finer q-grid can only find higher contact points, never lower."""
        torch.manual_seed(3)
        xyz = torch.randn(30, 3) * 2.0
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((30,), 0.29)
        coarse = afmize(xyz, _tip15(), radii, CONFIG_10)
        fine, _ = afmize_supersampled(xyz, radii, CONFIG_10, 2.0, 0.3, 15, factor=5)
        assert (fine >= coarse - 1e-9).all()
        assert (fine > coarse + 1e-3).any()

    def test_nested_factors_increase_the_image(self):
        """Refining a nested grid can only raise the image toward the continuum.

        Odd factors are not nested in general — factor 5's offsets are not a
        superset of factor 3's — so error need not fall monotonically as factor
        grows. When one factor divides another (1 | 3 | 9) the offset sets do
        nest, and the image is a max over that set, so it rises pointwise. The
        comparison is to float32 rounding of the surface, not exact: in float64
        the largest violation on this fixture is 2e-15.
        """
        torch.manual_seed(3)
        xyz = torch.randn(30, 3) * 2.0
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((30,), 0.29)
        args = (xyz, radii, CONFIG_10, 2.0, 0.3, 15)
        prev = None
        for factor in (1, 3, 9):
            img, _ = afmize_supersampled(*args, factor=factor)
            if prev is not None:
                assert (img >= prev - 1e-5).all(), f"factor {factor} lowered the image"
                assert (img > prev + 1e-3).any(), f"factor {factor} changed nothing"
            prev = img

    def test_factor_5_is_a_large_improvement(self):
        """Empirical regression on this fixture, not a general law.

        Guards the accuracy figures quoted in afmize_supersampled's docstring.
        """
        torch.manual_seed(4)
        xyz = torch.randn(30, 3) * 2.0
        xyz[:, 2] = xyz[:, 2].abs()
        radii = torch.full((30,), 0.29)
        args = (xyz, radii, CONFIG_10, 2.0, 0.3, 15)
        ref, _ = afmize_supersampled(*args, factor=9)
        mask = ref > 0
        err = lambda img: (img - ref)[mask].pow(2).mean().sqrt().item()
        err1 = err(afmize_supersampled(*args, factor=1)[0])
        err5 = err(afmize_supersampled(*args, factor=5)[0])
        assert err5 < err1 / 5, (err1, err5)

    @pytest.mark.parametrize("factor", [2, 4])
    def test_rejects_even_factor(self, factor):
        xyz = torch.tensor([[0.0, 0.0, 1.0]])
        radii = torch.tensor([0.5])
        with pytest.raises(ValueError):
            afmize_supersampled(xyz, radii, CONFIG_10, 2.0, 0.3, 15, factor=factor)

    def test_rejects_domain_not_divisible(self):
        # 7.5 pixels wide: ceil(7.5) = 8 coarse columns but ceil(22.5) = 23 fine
        # ones, so the coarse centers no longer land on fine centers
        config = dict(CONFIG_10, max_x=-2.5)
        xyz = torch.tensor([[0.0, 0.0, 1.0]])
        radii = torch.tensor([0.5])
        with pytest.raises(ValueError, match="not 3x the coarse grid"):
            afmize_supersampled(xyz, radii, config, 2.0, 0.3, 15, factor=3)


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
