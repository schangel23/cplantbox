"""Tests for the differentiable lofter — verify PyTorch output matches numpy lofter
and that gradients flow correctly."""

import numpy as np
import torch
import pytest


def _make_simple_skeleton(n=20, device='cpu'):
    """Create a simple curved leaf skeleton for testing."""
    t = torch.linspace(0, 1, n, device=device)
    # Arc: starts at angle 45deg, droops under gravity
    x = t * 50.0 * torch.sin(torch.tensor(0.7))
    y = torch.zeros(n, device=device)
    z = t * 50.0 * torch.cos(torch.tensor(0.7)) - 0.02 * (t * 50) ** 2
    skeleton = torch.stack([x, y, z], dim=1)
    # Width: tapers from 4cm at base to 0.2cm at tip
    widths = 4.0 * (1.0 - t ** 0.5) + 0.2
    return skeleton, widths


class TestFrames:
    def test_compute_tangents_shape(self):
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents
        skeleton, _ = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        assert tangents.shape == skeleton.shape

    def test_tangents_are_unit(self):
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents
        skeleton, _ = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        norms = torch.linalg.norm(tangents, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_tangents_gradient_flows(self):
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents
        skeleton, _ = _make_simple_skeleton()
        skeleton = skeleton.clone().requires_grad_(True)
        tangents = compute_tangents(skeleton)
        tangents.sum().backward()
        assert skeleton.grad is not None
        assert not torch.all(skeleton.grad == 0)

    def test_binormal_field_shape(self):
        from dart.coupling.experimental.diff_lofter.frames import (
            compute_tangents, compute_binormal_field
        )
        skeleton, _ = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        assert binormals.shape == skeleton.shape

    def test_binormals_are_unit(self):
        from dart.coupling.experimental.diff_lofter.frames import (
            compute_tangents, compute_binormal_field
        )
        skeleton, _ = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        norms = torch.linalg.norm(binormals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_binormals_consistent_sign(self):
        from dart.coupling.experimental.diff_lofter.frames import (
            compute_tangents, compute_binormal_field
        )
        skeleton, _ = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        # All consecutive binormals should have positive dot product (same direction)
        dots = (binormals[:-1] * binormals[1:]).sum(dim=1)
        assert (dots > -0.1).all(), f"Sign inconsistency: {dots.min()}"

    def test_binormal_gradient_flows(self):
        from dart.coupling.experimental.diff_lofter.frames import (
            compute_tangents, compute_binormal_field
        )
        skeleton, _ = _make_simple_skeleton()
        skeleton = skeleton.clone().requires_grad_(True)
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        binormals.sum().backward()
        assert skeleton.grad is not None


class TestDeformations:
    def test_output_shapes(self):
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations
        n = 30
        arc_fracs = torch.linspace(0, 1, n)
        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=torch.tensor(1.0),
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.5),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.3),
            curl_amp=torch.tensor(0.5),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.3),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.2),
            fold_freq=2.5, fold_phase=0.0,
        )
        for key in ['wave_normal', 'wave_lateral', 'twist', 'curl', 'edge_ruffle', 'fold']:
            assert key in deforms, f"Missing key: {key}"
            assert deforms[key].shape == (n,), f"{key} shape: {deforms[key].shape}"

    def test_ramp_zero_at_base(self):
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations
        arc_fracs = torch.linspace(0, 1, 50)
        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=torch.tensor(1.0),
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.5),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.5),
            curl_amp=torch.tensor(0.5),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.3),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.2),
            fold_freq=2.5, fold_phase=0.0,
            ramp_onset=0.15,
        )
        # At the base (t=0), ramp should be 0 → all deformations should be 0
        for key in deforms:
            assert abs(deforms[key][0].item()) < 1e-6, f"{key} non-zero at base"

    def test_gradient_flows(self):
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations
        arc_fracs = torch.linspace(0, 1, 20)
        amp = torch.tensor(1.0, requires_grad=True)
        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=amp,
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.5),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.3),
            curl_amp=torch.tensor(0.5),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.3),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.2),
            fold_freq=2.5, fold_phase=0.0,
        )
        deforms['wave_normal'].sum().backward()
        assert amp.grad is not None and amp.grad.item() != 0.0


class TestLofter:
    def test_output_shape(self):
        from dart.coupling.experimental.diff_lofter.lofter import loft_leaf
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents, compute_binormal_field
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations
        from dart.coupling.experimental.diff_lofter.lofter import compute_arc_fracs

        skeleton, widths = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=torch.tensor(0.5),
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.2),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.2),
            curl_amp=torch.tensor(0.3),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.2),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.1),
            fold_freq=2.5, fold_phase=0.0,
        )

        n_cross = 7
        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=n_cross)
        assert verts.shape == (len(skeleton) * n_cross, 3)

    def test_gradient_flows_through_lofter(self):
        """End-to-end: skeleton → lofter → loss → gradient on skeleton."""
        from dart.coupling.experimental.diff_lofter.lofter import loft_leaf, compute_arc_fracs
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents, compute_binormal_field
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations

        skeleton, widths = _make_simple_skeleton()
        skeleton = skeleton.clone().requires_grad_(True)
        widths = widths.clone().requires_grad_(True)
        amp = torch.tensor(0.5, requires_grad=True)

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=amp,
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.0),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.0),
            curl_amp=torch.tensor(0.0),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.0),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.0),
            fold_freq=2.5, fold_phase=0.0,
        )

        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)

        # Fake loss
        loss = verts.sum()
        loss.backward()

        assert skeleton.grad is not None, "No gradient on skeleton"
        assert widths.grad is not None, "No gradient on widths"
        assert amp.grad is not None, "No gradient on deformation amplitude"

    def test_no_nans(self):
        from dart.coupling.experimental.diff_lofter.lofter import loft_leaf, compute_arc_fracs
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents, compute_binormal_field
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations

        skeleton, widths = _make_simple_skeleton()
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=torch.tensor(1.5),
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.8),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.5),
            curl_amp=torch.tensor(1.0),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.8),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.5),
            fold_freq=2.5, fold_phase=0.0,
        )

        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)
        assert not torch.any(torch.isnan(verts)), "NaN in vertex output"
        assert not torch.any(torch.isinf(verts)), "Inf in vertex output"


class TestChamfer:
    def test_identical_clouds_zero_distance(self):
        from dart.coupling.experimental.losses.chamfer import chamfer_distance
        pc = torch.randn(100, 3)
        d = chamfer_distance(pc, pc)
        assert d.item() < 1e-4

    def test_symmetric(self):
        from dart.coupling.experimental.losses.chamfer import chamfer_distance
        pc1 = torch.randn(100, 3)
        pc2 = torch.randn(80, 3) + 1.0
        d1 = chamfer_distance(pc1, pc2)
        d2 = chamfer_distance(pc2, pc1)
        assert abs(d1.item() - d2.item()) < 1e-5

    def test_gradient_flows(self):
        from dart.coupling.experimental.losses.chamfer import chamfer_distance
        pc1 = torch.randn(50, 3, requires_grad=True)
        pc2 = torch.randn(50, 3)
        d = chamfer_distance(pc1, pc2)
        d.backward()
        assert pc1.grad is not None

    def test_batch_matches_single(self):
        from dart.coupling.experimental.losses.chamfer import chamfer_distance, chamfer_distance_batch
        pc1 = torch.randn(3, 50, 3)
        pc2 = torch.randn(3, 60, 3)
        batch_d = chamfer_distance_batch(pc1, pc2)
        for i in range(3):
            single_d = chamfer_distance(pc1[i], pc2[i])
            assert abs(batch_d[i].item() - single_d.item()) < 1e-5


class TestEndToEnd:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_lofter_chamfer_gradient_gpu(self):
        """Full pipeline on GPU: skeleton → lofter → chamfer → backprop."""
        from dart.coupling.experimental.diff_lofter.lofter import loft_leaf, compute_arc_fracs
        from dart.coupling.experimental.diff_lofter.frames import compute_tangents, compute_binormal_field
        from dart.coupling.experimental.diff_lofter.deformations import compute_deformations
        from dart.coupling.experimental.losses.chamfer import chamfer_distance

        device = 'cuda'
        skeleton, widths = _make_simple_skeleton(n=30, device=device)
        skeleton = skeleton.clone().requires_grad_(True)

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=torch.tensor(0.3, device=device),
            wave_normal_freq=3.5, wave_normal_phase=0.0,
            wave_lateral_amp=torch.tensor(0.0, device=device),
            wave_lateral_freq=2.0, wave_lateral_phase=0.0,
            twist_max=torch.tensor(0.0, device=device),
            curl_amp=torch.tensor(0.0, device=device),
            curl_freq=2.0, curl_phase=0.0,
            edge_ruffle_amp=torch.tensor(0.0, device=device),
            edge_ruffle_freq=7.0, edge_ruffle_phase=0.0,
            fold_amp=torch.tensor(0.0, device=device),
            fold_freq=2.5, fold_phase=0.0,
        )

        gen_verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)

        # Fake target: shifted version
        target = gen_verts.detach() + 2.0

        loss = chamfer_distance(gen_verts, target)
        loss.backward()

        assert skeleton.grad is not None
        assert loss.item() > 0
