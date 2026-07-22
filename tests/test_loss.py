import unittest

import torch
import torch.nn.functional as F

from coda.utils.loss import _compute_ce_loss, _masked_note_mse


def reference_grouped_ce(logits, counts, targets, valid, smoothing):
    losses = []
    correct = 0
    offset = 0
    for row, count in enumerate(counts):
        group = logits[offset:offset + count]
        offset += count
        target = int(targets[row])
        if count == 0 or not bool(valid[row]) or not 0 <= target < count:
            continue
        log_probs = F.log_softmax(group, dim=0)
        if smoothing > 0 and count > 1:
            target_distribution = torch.full_like(log_probs, smoothing / count)
            target_distribution[target] += 1.0 - smoothing
            losses.append(-(target_distribution * log_probs).sum())
        else:
            losses.append(-log_probs[target])
        correct += int(group.argmax() == target)
    return torch.stack(losses).mean(), correct, len(losses)


class VectorizedLossTests(unittest.TestCase):
    def test_grouped_ce_matches_reference_with_masks_and_smoothing(self):
        counts = [3, 1, 4, 0]
        targets = torch.tensor([2, 0, 3, 0])
        valid = torch.tensor([True, True, False, True])

        for smoothing in (0.0, 0.2):
            with self.subTest(smoothing=smoothing):
                logits_new = torch.tensor(
                    [0.2, -0.4, 1.1, 0.7, -0.3, 0.9, 0.1, -0.2],
                    requires_grad=True,
                )
                logits_reference = logits_new.detach().clone().requires_grad_(True)

                loss, correct, n_valid = _compute_ce_loss(
                    logits_new, counts, targets, valid, len(counts),
                    label_smoothing=smoothing,
                )
                expected_loss, expected_correct, expected_n = reference_grouped_ce(
                    logits_reference, counts, targets, valid, smoothing
                )

                torch.testing.assert_close(loss, expected_loss)
                self.assertEqual(int(correct), expected_correct)
                self.assertEqual(int(n_valid), expected_n)

                loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(logits_new.grad, logits_reference.grad)

    def test_masked_note_mse_matches_indexed_reference(self):
        predictions = torch.tensor(
            [[0.2, 0.4], [0.8, 0.1], [0.3, 0.7]], requires_grad=True
        )
        targets = torch.tensor([[0.1, 0.5], [0.5, 0.5], [0.2, 0.9]])
        valid = torch.tensor([True, False, True])

        actual = _masked_note_mse(predictions, targets, valid)
        expected = F.mse_loss(predictions[valid], targets[valid])
        torch.testing.assert_close(actual, expected)

    def test_all_invalid_loss_remains_differentiable_and_finite(self):
        logits = torch.tensor([1.0, -1.0], requires_grad=True)
        loss, correct, n_valid = _compute_ce_loss(
            logits, [2, 0], torch.tensor([-1, 0]), torch.tensor([True, True]), 2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(correct), 0)
        self.assertEqual(int(n_valid), 0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
