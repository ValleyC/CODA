import unittest

import torch

from coda.models.temporal import BarTransitionPrior, SystemTransitionPrior


class TransitionPriorTests(unittest.TestCase):
    def test_system_prior_is_normalized_and_splits_category_mass(self):
        prior = SystemTransitionPrior(
            same=0.0, forward_1=-1.0, backward_1=-1.5, far=-2.0
        )
        candidates = torch.tensor([0, 1, 2, 3, -1])

        probabilities = prior.log_prior(candidates, previous_position=0).exp()

        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=6)
        self.assertAlmostEqual(probabilities[2].item(), probabilities[3].item(), places=6)

        expected_category_mass = torch.softmax(torch.tensor([0.0, -1.0, -1.5, -2.0]), 0)
        observed_category_mass = torch.stack(
            (probabilities[0], probabilities[1], probabilities[4], probabilities[2:4].sum())
        )
        torch.testing.assert_close(observed_category_mass, expected_category_mass)

    def test_missing_previous_position_is_neutral(self):
        prior = SystemTransitionPrior()
        candidates = torch.arange(5)

        torch.testing.assert_close(prior.log_prior(candidates, None), torch.zeros(5))
        torch.testing.assert_close(prior.log_prior(candidates, -1), torch.zeros(5))

    def test_bar_categories_and_gradients(self):
        prior = BarTransitionPrior()
        candidates = torch.tensor([2, 3, 4, 1, 0, 6])

        log_probabilities = prior.log_prior(candidates, previous_position=2)
        loss = -log_probabilities[4]
        loss.backward()

        self.assertAlmostEqual(log_probabilities.exp().sum().item(), 1.0, places=6)
        self.assertIsNotNone(prior.relative_logits.grad)
        self.assertGreater(prior.relative_logits.grad.abs().sum().item(), 0.0)

    def test_reference_category_is_identifiable_anchor(self):
        prior = SystemTransitionPrior()
        self.assertEqual(tuple(prior.logits.shape), (4,))
        self.assertEqual(prior.logits[0].item(), 0.0)
        probabilities_before = prior.category_probabilities.clone()
        prior.set_logit("same", 1.0)
        self.assertGreater(prior.category_probabilities[0], probabilities_before[0])


if __name__ == "__main__":
    unittest.main()
