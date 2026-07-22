import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finalize_run.py"
SPEC = importlib.util.spec_from_file_location("finalize_run", SCRIPT)
finalize_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize_run)


class FinalizeRunTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(path, epoch, best_epoch):
        model_state = {"weight": torch.tensor([float(epoch)])}
        torch.save({
            "epoch": epoch,
            "best_epoch": best_epoch,
            "args": {
                "num_epochs": finalize_run.EXPECTED_PHASE2_EPOCHS,
                **finalize_run.EXPECTED_PHASE2_SCHEDULE,
            },
            "model_state_dict": model_state,
            "criterion_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "scaler_state_dict": {},
            "rng_state": {},
        }, path)

    def _fixture(self, root):
        phase1 = root / "phase1"
        phase2 = root / "phase2"
        results = root / "results"
        run_root = root / "run"
        phase1.mkdir()
        phase2.mkdir()
        results.mkdir()
        (phase1 / "best_checkpoint.pt").write_bytes(b"phase1")
        best_epoch = finalize_run.EXPECTED_PHASE2_EPOCHS - 1
        for epoch in range(finalize_run.EXPECTED_PHASE2_EPOCHS):
            self._write_checkpoint(
                phase2 / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch,
                best_epoch,
            )
        self._write_checkpoint(
            phase2 / "best_checkpoint.pt", best_epoch, best_epoch
        )
        torch.save(
            {"weight": torch.tensor([float(best_epoch)])},
            phase2 / "best_model.pt",
        )
        for name, count in finalize_run.EXPECTED_EVALUATION_COUNTS.items():
            (results / f"{name}_summary.json").write_text(json.dumps({
                "complete": True,
                "requested_pieces": count,
                "successful_pieces": count,
                "failed_pieces": [],
            }))
        jump_manifest = root / "jump_manifest.json"
        jump_manifest.write_text(json.dumps({
            "repeat_pieces": 66,
            "random_pieces": 28,
            "random_piece_seeds": {str(i): i for i in range(28)},
        }))
        test_log = root / "tests.log"
        test_log.write_text("Ran 32 tests in 0.5s\n\nOK\n")
        return phase1, phase2, results, run_root, jump_manifest, test_log

    def test_complete_evidence_writes_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            path, manifest = finalize_run.finalize_run("run", *values)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["tests_passed"], 32)
            self.assertEqual(set(manifest["evaluations"]), {
                "standard", "repeat", "random"
            })
            self.assertEqual(len(manifest["artifacts"]["final_model"]["sha256"]), 64)
            self.assertEqual(
                manifest["training"]["phase2"]["completed_epochs"], 20
            )
            self.assertEqual(
                len([
                    key for key in manifest["artifacts"]
                    if key.startswith("phase2_checkpoint_epoch_")
                ]),
                20,
            )
            self.assertTrue(Path(path).is_file())

    def test_incomplete_evaluation_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = self._fixture(root)
            summary = values[2] / "standard_summary.json"
            data = json.loads(summary.read_text())
            data["complete"] = False
            summary.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                finalize_run.finalize_run("run", *values)
            self.assertFalse((values[3] / "run_manifest.json").exists())

    def test_early_stopped_run_records_phase1_model(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            phase1, _, results, run_root, jump_manifest, test_log = values
            final_model = phase1 / "best_model.pt"
            final_model.write_bytes(b"early-stopped-model")
            path, manifest = finalize_run.finalize_run(
                "run", phase1, None, results, run_root, jump_manifest,
                test_log, final_model_path=final_model,
                early_stop_reason="streaming validation converged",
            )
            self.assertTrue(Path(path).is_file())
            self.assertTrue(manifest["training"]["early_stopped"])
            self.assertFalse(manifest["training"]["phase2_completed"])
            self.assertIsNone(manifest["phase2_dir"])
            self.assertNotIn(
                "phase2_best_checkpoint", manifest["artifacts"]
            )
            self.assertEqual(
                manifest["artifacts"]["final_model"]["path"],
                str(final_model.resolve()),
            )

    def test_missing_phase2_requires_early_stop_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            phase1, _, results, run_root, jump_manifest, test_log = values
            with self.assertRaisesRegex(RuntimeError, "early-stop reason"):
                finalize_run.finalize_run(
                    "run", phase1, None, results, run_root, jump_manifest,
                    test_log,
                )

    def test_missing_phase2_epoch_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            (values[1] / "checkpoint_epoch_019.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing checkpoint_epoch"):
                finalize_run.finalize_run("run", *values)
            self.assertFalse((values[3] / "run_manifest.json").exists())

    def test_extra_phase2_epoch_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            self._write_checkpoint(
                values[1] / "checkpoint_epoch_020.pt", 20, 19
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                finalize_run.finalize_run("run", *values)
            self.assertFalse((values[3] / "run_manifest.json").exists())

    def test_wrong_phase2_schedule_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            path = values[1] / "checkpoint_epoch_007.pt"
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["args"]["ss_max_p"] = 0.5
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "ss_max_p"):
                finalize_run.finalize_run("run", *values)
            self.assertFalse((values[3] / "run_manifest.json").exists())

    def test_nonfinite_phase2_checkpoint_cannot_write_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self._fixture(Path(directory))
            path = values[1] / "checkpoint_epoch_012.pt"
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["model_state_dict"]["weight"][0] = float("nan")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                finalize_run.finalize_run("run", *values)
            self.assertFalse((values[3] / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
