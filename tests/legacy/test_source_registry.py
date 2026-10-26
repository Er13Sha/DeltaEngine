import tempfile
import unittest
from pathlib import Path

from logical_delta_matter.source_registry import SourceRegistry, sha256_file


class SourceRegistryTests(unittest.TestCase):
    def test_registers_and_verifies_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fact.txt"
            path.write_text("strict fact", encoding="utf-8")
            registry = SourceRegistry()
            record = registry.register_file("fact.1", path)
            self.assertEqual(record.sha256, sha256_file(path))
            self.assertTrue(registry.verify("fact.1"))

    def test_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fact.txt"
            path.write_text("strict fact", encoding="utf-8")
            registry = SourceRegistry()
            registry.register_file("fact.1", path)
            path.write_text("changed", encoding="utf-8")
            self.assertFalse(registry.verify("fact.1"))


if __name__ == "__main__":
    unittest.main()
