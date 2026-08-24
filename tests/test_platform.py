from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.audit_log import AuditLog
from src.data_repository import DataRepository
from src.optimizer import DispatchOptimizer
from src.policy_engine import PolicyEngine
from src.risk_engine import RiskEngine


ROOT = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads((ROOT / "data" / "platform_data.json").read_text(encoding="utf-8"))

    def test_repository_filters(self) -> None:
        repository = DataRepository(ROOT / "data" / "platform_data.json")
        rows = repository.watchlist(road="G9411")
        self.assertTrue(rows)
        self.assertTrue(all(row["road_code"] == "G9411" for row in rows))

    def test_real_student_model_replays_known_case(self) -> None:
        engine = RiskEngine(ROOT / "models", self.data["default_features"])
        result = engine.evaluate(self.data["default_features"])
        self.assertTrue(engine.status["ready"])
        self.assertGreaterEqual(result["score"], 0.89)
        self.assertTrue(result["alert"])

    def test_policy_review_is_human_gated(self) -> None:
        engine = PolicyEngine(self.data["policy_library"])
        result = engine.review(
            {
                "score": 0.91,
                "median_speed_km_h": 23.8,
                "upstream_traffic": 142,
                "downstream_traffic": 101,
                "weather": "Overcast",
                "type_hypothesis": "翻车",
            }
        )
        self.assertTrue(result["human_review_required"])
        self.assertTrue(result["actions"])
        self.assertTrue(all(item["authorized"] for item in result["actions"]))

    def test_optimizer_respects_capacities(self) -> None:
        optimizer = DispatchOptimizer()
        result = optimizer.allocate(
            self.data["watchlist"],
            {"warning_capacity": 4, "patrol_capacity": 2, "emergency_capacity": 1},
        )
        self.assertLessEqual(result["counts"]["warning"], 4)
        self.assertLessEqual(result["counts"]["patrol"], 2)
        self.assertLessEqual(result["counts"]["emergency"], 1)

    def test_audit_log_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = AuditLog(Path(directory))
            record = audit.append({"action": "确认", "target": "G9411_1_137370"})
            self.assertEqual(audit.list()[0]["id"], record["id"])


if __name__ == "__main__":
    unittest.main()

