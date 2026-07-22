"""Normalized, learnable transition priors for hierarchical score tracking."""

from collections import OrderedDict
from typing import Mapping, Sequence

import torch
import torch.nn as nn


class CategoricalTransitionPrior(nn.Module):
    """A proper categorical transition distribution over available candidates.

    Each candidate belongs to exactly one transition category. Learned category
    mass is divided uniformly among candidates in that category and then
    renormalized over categories that are present. The first category is the
    reference category with a fixed logit of zero, removing the otherwise
    unidentifiable additive degree of freedom.
    """

    def __init__(self, category_names: Sequence[str], initial_logits: Sequence[float]):
        super().__init__()
        if len(category_names) < 2 or len(category_names) != len(initial_logits):
            raise ValueError("category_names and initial_logits must have the same length >= 2")

        self.category_names = tuple(category_names)
        initial = torch.as_tensor(initial_logits, dtype=torch.float32)
        relative = initial[1:] - initial[0]
        self.relative_logits = nn.Parameter(relative)
        # Transition logits are scale parameters, not ordinary weights.
        self.relative_logits._no_weight_decay = True

    @property
    def logits(self) -> torch.Tensor:
        zero = self.relative_logits.new_zeros(1)
        return torch.cat((zero, self.relative_logits), dim=0)

    @property
    def category_probabilities(self) -> torch.Tensor:
        return self.logits.softmax(dim=0)

    def logits_dict(self) -> Mapping[str, float]:
        values = self.logits.detach().cpu().tolist()
        return OrderedDict(zip(self.category_names, values))

    def probabilities_dict(self) -> Mapping[str, float]:
        values = self.category_probabilities.detach().cpu().tolist()
        return OrderedDict(zip(self.category_names, values))

    def set_logit(self, category: str, value: float) -> None:
        """Set a relative category logit for evaluation-time overrides."""
        if category not in self.category_names:
            raise KeyError(f"Unknown transition category: {category}")
        index = self.category_names.index(category)
        if index == 0:
            # Shifting every other relative logit by -value is equivalent to
            # assigning the requested absolute value to the anchored category.
            with torch.no_grad():
                self.relative_logits.sub_(float(value))
            return
        with torch.no_grad():
            self.relative_logits[index - 1].fill_(float(value))

    def category_ids(self, deltas: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prior(self, candidate_positions: torch.Tensor, previous_position) -> torch.Tensor:
        """Return normalized log transition probabilities for one candidate set.

        A missing previous position yields a neutral all-zero bias. This is
        preferable to injecting an arbitrary prior at piece/page initialization.
        """
        if candidate_positions.ndim != 1:
            raise ValueError("candidate_positions must be one-dimensional")
        if candidate_positions.numel() == 0:
            return self.relative_logits.new_zeros(0)

        if previous_position is None:
            return self.relative_logits.new_zeros(candidate_positions.numel())
        if torch.is_tensor(previous_position):
            if previous_position.numel() != 1:
                raise ValueError("previous_position must be scalar")
            previous_position = previous_position.to(
                device=candidate_positions.device, dtype=candidate_positions.dtype
            )
            if previous_position.item() < 0:
                return self.relative_logits.new_zeros(candidate_positions.numel())
        elif previous_position < 0:
            return self.relative_logits.new_zeros(candidate_positions.numel())

        deltas = candidate_positions - previous_position
        categories = self.category_ids(deltas)
        category_logits = self.logits.to(candidate_positions.device)
        counts = torch.bincount(categories, minlength=len(self.category_names)).clamp_min(1)

        # Give each category its learned mass regardless of how many candidates
        # happen to fall into that category on the current score page.
        candidate_logits = category_logits[categories] - counts[categories].log()
        return candidate_logits - torch.logsumexp(candidate_logits, dim=0)


class SystemTransitionPrior(CategoricalTransitionPrior):
    """Directional transition prior over page-local system indices."""

    def __init__(self, same: float = 0.0, forward_1: float = -1.0,
                 backward_1: float = -2.0, far: float = -5.0):
        super().__init__(
            ("same", "forward_1", "backward_1", "far"),
            (same, forward_1, backward_1, far),
        )

    def category_ids(self, deltas: torch.Tensor) -> torch.Tensor:
        categories = torch.full_like(deltas, 3, dtype=torch.long)
        categories = torch.where(deltas == 0, torch.zeros_like(categories), categories)
        categories = torch.where(deltas == 1, torch.ones_like(categories), categories)
        categories = torch.where(deltas == -1, torch.full_like(categories, 2), categories)
        return categories


class BarTransitionPrior(CategoricalTransitionPrior):
    """Transition prior over page-local bar indices."""

    def __init__(self, stay: float = 0.0, forward_1: float = -0.3,
                 forward_2: float = -1.5, backward_1: float = -2.0,
                 far: float = -5.0):
        super().__init__(
            ("stay", "forward_1", "forward_2", "backward_1", "far"),
            (stay, forward_1, forward_2, backward_1, far),
        )

    def category_ids(self, deltas: torch.Tensor) -> torch.Tensor:
        categories = torch.full_like(deltas, 4, dtype=torch.long)
        categories = torch.where(deltas == 0, torch.zeros_like(categories), categories)
        categories = torch.where(deltas == 1, torch.ones_like(categories), categories)
        categories = torch.where(deltas == 2, torch.full_like(categories, 2), categories)
        categories = torch.where(deltas == -1, torch.full_like(categories, 3), categories)
        return categories
