import importlib.util
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_batch.py"
SPEC = importlib.util.spec_from_file_location("evaluate_batch", SCRIPT)
evaluate_batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_batch)


class EvaluateBatchOutputTests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            param_path="model.pt",
            test_dir="test-data",
            with_video=False,
            video_dir="videos",
            benchmark=False,
            break_mode=False,
            break_onset_threshold=None,
            break_release_threshold=None,
            break_silence_onset=None,
            break_grace_frames=None,
            break_prior_scale=None,
            break_beam_k=None,
            break_beam_m=None,
        )

    def test_piece_evaluation_publishes_only_fresh_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "piece.json"
            final_path.write_text(json.dumps({"stale": True}), encoding="utf-8")

            def successful_child(cmd, **_kwargs):
                staging_path = Path(cmd[cmd.index("--save_metrics") + 1])
                staging_path.write_text(
                    json.dumps({"sys_accuracy": 0.75}), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="")

            with patch.object(
                evaluate_batch.subprocess, "run", side_effect=successful_child
            ):
                metrics = evaluate_batch.run_evaluate(
                    "piece", self._args(), str(final_path)
                )

            self.assertEqual(metrics, {"sys_accuracy": 0.75})
            self.assertEqual(json.loads(final_path.read_text()), metrics)
            self.assertEqual(list(final_path.parent.glob("piece.json.tmp-*")), [])

    def test_stale_metrics_are_not_reused_when_child_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "piece.json"
            final_path.write_text(json.dumps({"stale": True}), encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout="")

            with patch.object(
                evaluate_batch.subprocess, "run", return_value=completed
            ):
                captured = io.StringIO()
                with redirect_stdout(captured):
                    metrics = evaluate_batch.run_evaluate(
                        "piece", self._args(), str(final_path)
                    )

            self.assertIsNone(metrics)
            self.assertIn("produced no metrics file", captured.getvalue())
            self.assertEqual(
                json.loads(final_path.read_text()), {"stale": True}
            )
            self.assertEqual(list(final_path.parent.glob("piece.json.tmp-*")), [])

    def test_summary_records_completeness_and_numpy_scalars(self):
        complete = evaluate_batch.build_serializable_summary(
            {"n_pieces": 3, "mean_sys_accuracy": np.float64(0.75)},
            "run", [], 3,
        )
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["requested_pieces"], 3)
        self.assertEqual(complete["successful_pieces"], 3)
        self.assertIsInstance(complete["mean_sys_accuracy"], float)

        incomplete = evaluate_batch.build_serializable_summary(
            {"n_pieces": 2}, "run", ["broken_piece"], 3,
        )
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["failed_pieces"], ["broken_piece"])

    def test_atomic_json_dump_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            evaluate_batch.atomic_json_dump({"complete": True}, str(path))
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertEqual(list(path.parent.glob("summary.json.tmp-*")), [])


if __name__ == "__main__":
    unittest.main()
