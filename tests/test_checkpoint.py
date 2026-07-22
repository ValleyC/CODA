import os
import tempfile
import unittest

import torch
import torch.nn as nn

from coda.utils.checkpoint import (
    atomic_torch_save,
    extract_model_state,
    make_training_checkpoint,
    validate_resume_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def test_partial_epoch_checkpoint_is_diagnostic_only(self):
        checkpoint = {
            "model_state_dict": {"weight": torch.ones(1)},
            "partial_epoch": True,
            "resume_from_completed_checkpoint": "/run/latest_checkpoint.pt",
        }
        with self.assertRaisesRegex(
            ValueError, "would duplicate optimizer updates"
        ):
            validate_resume_checkpoint(checkpoint, "failure_checkpoint.pt")

        checkpoint["partial_epoch"] = False
        validate_resume_checkpoint(checkpoint, "latest_checkpoint.pt")

    def test_atomic_save_and_structured_model_extraction(self):
        model = nn.Linear(3, 2)
        criterion = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        checkpoint = make_training_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=4,
            min_loss=0.5,
            best_streaming_bar_acc=0.75,
            best_epoch=3,
            metrics={"val_loss": 0.25, "streaming": {"bar_acc": 0.4}},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            atomic_torch_save(checkpoint, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(loaded["epoch"], 4)
        self.assertEqual(loaded["metrics"]["val_loss"], 0.25)
        self.assertEqual(loaded["metrics"]["streaming"]["bar_acc"], 0.4)
        self.assertEqual(
            set(loaded["criterion_state_dict"]), set(criterion.state_dict())
        )
        for name, value in criterion.state_dict().items():
            self.assertTrue(torch.equal(loaded["criterion_state_dict"][name], value))
        self.assertEqual(set(extract_model_state(loaded)), set(model.state_dict()))

    def test_data_parallel_prefix_is_removed(self):
        state = {"module.weight": torch.ones(1), "module.bias": torch.zeros(1)}
        extracted = extract_model_state(state)
        self.assertEqual(set(extracted), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
