from types import SimpleNamespace
from types import ModuleType
import sys
import unittest

import torch
import torch.nn as nn

# The unit test exercises only the pure training loop; keep it runnable in a
# minimal CPU test environment where the optional TensorBoard package is not
# installed.
tensorboard_stub = ModuleType('torch.utils.tensorboard')
tensorboard_stub.SummaryWriter = object
sys.modules.setdefault('torch.utils.tensorboard', tensorboard_stub)

from scripts.train import iterate_selection


class _ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, score, **kwargs):
        batch_size = score.shape[0]
        value = score.mean() * self.weight
        z = value.expand(batch_size, 1)
        audio_seq = value.expand(batch_size, 1, 1)
        return {'z': z, 'audio_seq': audio_seq}


class _ToyCriterion(nn.Module):
    def __init__(self, nonfinite_metric=False, python_metric=None):
        super().__init__()
        self.nonfinite_metric = nonfinite_metric
        self.python_metric = python_metric

    def forward(self, outputs, *args, **kwargs):
        loss = outputs['z'].mean() + outputs['audio_seq'].mean()
        accuracy = loss.detach() * 0 + 0.5
        if self.nonfinite_metric:
            accuracy = accuracy * float('nan')
        result = {'loss': loss, 'sys_loss': loss.detach(), 'sys_acc': accuracy}
        if self.python_metric is not None:
            result['python_metric'] = self.python_metric
        return result


def _toy_batch():
    return SimpleNamespace(
        scores=torch.ones(2, 1, 2, 2),
        perf=[torch.ones(8), torch.ones(6)],
        gt_system_idx=torch.zeros(2, dtype=torch.long),
        gt_bar_in_sys=torch.zeros(2, dtype=torch.long),
        gt_note_position=torch.zeros(2, 2),
        gt_valid=torch.ones(2, dtype=torch.bool),
        prev_gt_system_idx=torch.full((2,), -1, dtype=torch.long),
        prev_gt_bar_page_idx=torch.full((2,), -1, dtype=torch.long),
        system_boxes=[torch.empty(0, 4), torch.empty(0, 4)],
        bar_boxes=[torch.empty(0, 4), torch.empty(0, 4)],
        bars_per_system=[[], []],
        file_names=['first', 'second'],
    )


class TrainingMetricSynchronizationTests(unittest.TestCase):
    def test_training_metrics_are_returned_and_the_step_is_applied(self):
        model = _ToyNetwork()
        criterion = _ToyCriterion()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()

        stats = iterate_selection(
            model,
            [_toy_batch()],
            criterion,
            optimizer=optimizer,
            clip_grads=1.0,
            device=torch.device('cpu'),
        )

        self.assertAlmostEqual(stats['loss'], 2.0)
        self.assertAlmostEqual(stats['sys_acc'], 0.5)
        self.assertFalse(torch.equal(model.weight, before))

    def test_nonfinite_metric_prevents_the_training_update(self):
        model = _ToyNetwork()
        criterion = _ToyCriterion(nonfinite_metric=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()

        with self.assertRaisesRegex(FloatingPointError, 'Non-finite batch 0'):
            iterate_selection(
                model,
                [_toy_batch()],
                criterion,
                optimizer=optimizer,
                clip_grads=1.0,
                device=torch.device('cpu'),
            )

        self.assertTrue(torch.equal(model.weight, before))
        self.assertIsNone(model.weight.grad)

    def test_nonfinite_python_metric_also_prevents_the_training_update(self):
        model = _ToyNetwork()
        criterion = _ToyCriterion(python_metric=float('nan'))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()

        with self.assertRaisesRegex(FloatingPointError, 'python_metric'):
            iterate_selection(
                model,
                [_toy_batch()],
                criterion,
                optimizer=optimizer,
                clip_grads=1.0,
                device=torch.device('cpu'),
            )

        self.assertTrue(torch.equal(model.weight, before))
        self.assertIsNone(model.weight.grad)


if __name__ == '__main__':
    unittest.main()
