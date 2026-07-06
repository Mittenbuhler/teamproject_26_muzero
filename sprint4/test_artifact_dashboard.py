import json
import tempfile
import unittest
from pathlib import Path

from build_artifact_dashboard import generate_dashboard
from dashboard_server import rename_run, update_folders


class ArtifactDashboardTest(unittest.TestCase):
    def test_dashboard_only_embeds_state_report_with_clickable_raw_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifacts"
            nested_dir = artifact_dir / "diagnostics"
            nested_dir.mkdir(parents=True)
            (artifact_dir / "state_mcts_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "runs": [
                            {
                                "run_id": "baseline",
                                "status": "completed",
                                "selected_models": [],
                                "training_metrics": {},
                                "evaluations": {
                                    "baseline": {
                                        "mean_reward": 100.0,
                                        "episode_rewards": [90.0, 110.0],
                                    }
                                },
                                "settings": {
                                    "search_depth": 5,
                                    "note": "</script><unsafe>",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (nested_dir / "checks.json").write_text(
                json.dumps({"passed": False, "note": "</script><unsafe>"}),
                encoding="utf-8",
            )
            output = artifact_dir / "diagnostics_dashboard.html"
            count = generate_dashboard(artifact_dir, output)
            html = output.read_text(encoding="utf-8")

            self.assertEqual(count, 1)
            self.assertIn("state_mcts_report.json", html)
            self.assertNotIn("diagnostics/checks.json", html)
            self.assertIn("Complete state_mcts_report.json", html)
            self.assertIn("State MCTS experiments", html)
            self.assertIn("Reward graphs", html)
            self.assertIn("automatically sorted alphabetically", html)
            self.assertIn("raw-run-${runIndex}", html)
            self.assertIn("openRaw", html)
            self.assertIn("localeCompare", html)
            self.assertNotIn("dropRun", html)
            self.assertNotIn("/api/run-order", html)
            self.assertIn("/api/folders", html)
            self.assertIn("Folder organization", html)
            self.assertIn("Editable run names and folder assignments", html)
            self.assertIn("saveRunName", html)
            self.assertIn("groupedTiles", html)
            self.assertIn("function folderPanel", html)
            self.assertIn('<details class="folder-block" data-folder-key=', html)
            self.assertIn("Click any folder heading to expand or collapse it", html)
            self.assertIn("FOLDER_OPEN_STORAGE_KEY", html)
            self.assertIn("rememberFolderState", html)
            self.assertIn("Bulk run actions", html)
            self.assertIn("selectedRunIds", html)
            self.assertIn("bulkAssignSelected", html)
            self.assertIn("data-select-run", html)
            self.assertNotIn("</script><unsafe>", html)
            self.assertNotIn("https://", html)

    def test_rename_persists_display_name_without_changing_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "state_mcts_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "runs": [
                            {
                                "run_id": "stable-id",
                                "status": "completed",
                                "evaluations": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            renamed = rename_run(report_path, "stable-id", "Depth 15 dynamics")
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(renamed["display_name"], "Depth 15 dynamics")
            self.assertEqual(stored["runs"][0]["run_id"], "stable-id")
            self.assertEqual(stored["runs"][0]["display_name"], "Depth 15 dynamics")

            rename_run(report_path, "stable-id", "")
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn("display_name", stored["runs"][0])

    def test_folder_metadata_persists_order_and_run_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "state_mcts_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "runs": [
                            {"run_id": "depth-10"},
                            {"run_id": "depth-30"},
                            {"run_id": "new-run"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            folders = [
                {"id": "comparison", "name": "Depth comparison"},
                {"id": "references", "name": "References"},
            ]
            assignments = {
                "depth-10": "comparison",
                "depth-30": "comparison",
            }
            stored_folders, stored_assignments = update_folders(
                report_path, folders, assignments
            )
            stored = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(stored_folders, folders)
            self.assertEqual(stored_assignments, assignments)
            self.assertEqual(stored["dashboard_folders"], folders)
            self.assertEqual(stored["dashboard_run_folders"], assignments)
            self.assertNotIn("new-run", stored["dashboard_run_folders"])

            with self.assertRaises(KeyError):
                update_folders(
                    report_path,
                    folders,
                    {"new-run": "missing-folder"},
                )


if __name__ == "__main__":
    unittest.main()
