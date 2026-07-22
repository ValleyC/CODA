import unittest

import torch
import torch.nn as nn

from coda.utils.optim import (
    build_adamw,
    clip_gradient_at_boundary,
    guarded_optimizer_step,
)


class OptimizerCoverageTests(unittest.TestCase):
    def test_boundary_gradient_clip_handles_huge_and_nonfinite_values(self):
        gradient = torch.tensor([2e38, -2e38, float('nan'), float('inf')])
        clipped = clip_gradient_at_boundary(gradient, max_norm=1.0)

        self.assertTrue(torch.isfinite(clipped).all())
        self.assertLessEqual(float(clipped.double().norm()), 1.000001)
        self.assertGreater(float(clipped[:2].abs().sum()), 0.0)

    def test_boundary_gradient_clip_leaves_small_gradient_unchanged(self):
        gradient = torch.tensor([0.1, -0.2, 0.3])
        clipped = clip_gradient_at_boundary(gradient, max_norm=1.0)
        self.assertTrue(torch.equal(clipped, gradient))

    def test_guarded_step_updates_and_clips_finite_gradients(self):
        model = nn.Linear(2, 1, bias=False)
        criterion = nn.Module()
        criterion.log_scale = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(criterion.parameters()), lr=0.1
        )

        before = model.weight.detach().clone()
        loss = model(torch.full((4, 2), 100.0)).sum() * criterion.log_scale
        loss.backward()
        (
            stepped,
            total_norm,
            bad_gradients,
            bad_count,
            recovered_norm_overflow,
            host_metric_values,
        ) = guarded_optimizer_step(model, criterion, optimizer, clip_grads=0.25)

        clipped_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().float().norm()
            for parameter in list(model.parameters()) + list(criterion.parameters())
        ]))
        self.assertTrue(stepped)
        self.assertTrue(torch.isfinite(torch.tensor(total_norm)))
        self.assertGreater(total_norm, 0.25)
        self.assertLessEqual(float(clipped_norm), 0.250001)
        self.assertEqual(bad_gradients, {})
        self.assertEqual(bad_count, 0)
        self.assertFalse(recovered_norm_overflow)
        self.assertIsNone(host_metric_values)
        self.assertFalse(torch.equal(model.weight, before))

    def test_guarded_step_returns_metrics_from_the_norm_synchronization(self):
        model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.ones(2, 2)).sum().backward()

        result = guarded_optimizer_step(
            model,
            nn.Identity(),
            optimizer,
            clip_grads=1.0,
            metric_values=torch.tensor([1.25, -0.5]),
        )

        self.assertTrue(result[0])
        self.assertEqual(result[-1], [1.25, -0.5])

    def test_guarded_step_rejects_nonfinite_metrics_before_update(self):
        model = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        model(torch.ones(2, 2)).sum().backward()

        (
            stepped,
            _,
            bad_gradients,
            bad_count,
            recovered_norm_overflow,
            host_metric_values,
        ) = guarded_optimizer_step(
            model,
            nn.Identity(),
            optimizer,
            clip_grads=1.0,
            metric_values=torch.tensor([1.0, float('nan')]),
        )

        self.assertFalse(stepped)
        self.assertEqual(bad_gradients, {})
        self.assertEqual(bad_count, 0)
        self.assertFalse(recovered_norm_overflow)
        self.assertEqual(host_metric_values[0], 1.0)
        self.assertTrue(torch.isnan(torch.tensor(host_metric_values[1])))
        self.assertTrue(torch.equal(model.weight, before))
        self.assertIsNone(model.weight.grad)

    def test_guarded_step_rejects_nonfinite_model_and_criterion_gradients(self):
        model = nn.Linear(2, 1, bias=False)
        criterion = nn.Module()
        criterion.log_scale = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()), lr=0.1
        )
        before_model = model.weight.detach().clone()
        before_criterion = criterion.log_scale.detach().clone()

        model.weight.grad = torch.tensor([[float('inf'), 1.0]])
        criterion.log_scale.grad = torch.tensor([float('nan')])
        (
            stepped,
            total_norm,
            bad_gradients,
            bad_count,
            recovered_norm_overflow,
            host_metric_values,
        ) = guarded_optimizer_step(model, criterion, optimizer, clip_grads=1.0)

        self.assertFalse(stepped)
        self.assertFalse(torch.isfinite(torch.tensor(total_norm)))
        self.assertIn('model.weight', bad_gradients)
        self.assertIn('criterion.log_scale', bad_gradients)
        self.assertGreaterEqual(bad_count, 2)
        self.assertFalse(recovered_norm_overflow)
        self.assertIsNone(host_metric_values)
        self.assertTrue(torch.equal(model.weight, before_model))
        self.assertTrue(torch.equal(criterion.log_scale, before_criterion))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertIsNone(criterion.log_scale.grad)
        self.assertEqual(len(optimizer.state), 0)

    def test_guarded_step_recovers_finite_float32_norm_overflow(self):
        model = nn.Linear(4, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        model.weight.grad = torch.full_like(model.weight, 2e38)

        (
            stepped,
            total_norm,
            bad_gradients,
            bad_count,
            recovered_norm_overflow,
            host_metric_values,
        ) = guarded_optimizer_step(
            model, nn.Identity(), optimizer, clip_grads=1.0
        )

        self.assertTrue(stepped)
        self.assertTrue(recovered_norm_overflow)
        self.assertTrue(torch.isfinite(torch.tensor(total_norm, dtype=torch.float64)))
        self.assertGreater(total_norm, torch.finfo(torch.float32).max)
        self.assertEqual(bad_gradients, {})
        self.assertEqual(bad_count, 0)
        self.assertIsNone(host_metric_values)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertFalse(torch.equal(model.weight, before))
        self.assertLessEqual(float(model.weight.grad.float().norm()), 1.000001)

    def test_direct_and_multihead_attention_parameters_are_optimized(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.direct_scale = nn.Parameter(torch.tensor([1.0, 2.0]))
                self.attention = nn.MultiheadAttention(4, 2, batch_first=True)
                self.projection = nn.Linear(4, 1)

        model = ToyModel()
        optimizer, summary = build_adamw(model, learning_rate=0.1, weight_decay=0.01)

        optimized_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected_ids = {id(parameter) for parameter in model.parameters()}
        self.assertEqual(optimized_ids, expected_ids)
        self.assertEqual(summary["total_parameters"], sum(p.numel() for p in model.parameters()))

        direct_before = model.direct_scale.detach().clone()
        attention_before = model.attention.in_proj_weight.detach().clone()
        query = torch.randn(2, 3, 4)
        attended, _ = model.attention(query, query, query)
        loss = model.projection(attended).sum() + model.direct_scale.sum()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(model.direct_scale, direct_before))
        self.assertFalse(torch.equal(model.attention.in_proj_weight, attention_before))

if __name__ == "__main__":
    unittest.main()
