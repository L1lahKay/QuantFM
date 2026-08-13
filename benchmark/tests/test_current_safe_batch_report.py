from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmark" / "report" / "generate_current_safe_batch_report.py"


def load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("current_safe_batch_report", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = load_module()


class CurrentSafeBatchReportTests(unittest.TestCase):
    def test_exact_batch_generates_standalone_report_and_four_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report"
            result = REPORT.generate(output)
            self.assertEqual(result["run_count"], 45)
            self.assertEqual(result["cell_count"], 15)
            self.assertEqual(len(result["images"]), 4)
            document = output / "CURRENT_SAFE_N3_EXPERIMENT_REPORT.md"
            markdown = document.read_text(encoding="utf-8")
            self.assertIn("45/45", markdown)
            self.assertIn("images/training-time-n3.svg", markdown)
            self.assertIn("一次性 current-safe 授权已经用完", markdown)
            self.assertLess(
                markdown.index("## 1. 结论"), markdown.index("## 2. 实验设计")
            )
            self.assertLess(
                markdown.index("## 2. 实验设计"), markdown.index("## 3. 实验结果")
            )
            self.assertNotIn("独立实验报告；", markdown)
            for link in re.findall(r"\]\(([^)]+)\)", markdown):
                self.assertTrue((document.parent / link).resolve().is_file(), link)
            payload = json.loads(
                (output / "current-safe-20260806-results.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["run_count"], 45)
            self.assertEqual(payload["cell_count"], 15)
            self.assertEqual(len(payload["cells"]), 15)
            for image in (output / "images").glob("*.svg"):
                svg = image.read_text(encoding="utf-8")
                self.assertIn("横线：最小值–最大值", svg)
                self.assertNotIn("N/A", svg)
                root = ElementTree.parse(image).getroot()
                self.assertEqual(root.attrib["viewBox"], "0 0 1280 690")


if __name__ == "__main__":
    unittest.main()
