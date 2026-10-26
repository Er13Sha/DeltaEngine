"""Охрана границы доверия (docs/delta-engine-plan.md, раздел 3).

delta_kernel — доверенное ядро: только стандартная библиотека, только точная арифметика,
без импорта остальных пакетов проекта. Пока ядро пусто, проверки проходят тривиально;
они начнут работать сразу, как только появится код (этап M1).
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "delta_kernel"


def kernel_files() -> list[Path]:
    return sorted(p for p in KERNEL.rglob("*.py") if "__pycache__" not in p.parts)


def parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def top_level_imports(path: Path):
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


class TrustBoundaryTests(unittest.TestCase):
    def test_packages_are_importable(self):
        import adapters  # noqa: F401
        import delta_app  # noqa: F401
        import delta_kernel  # noqa: F401
        import delta_tools  # noqa: F401

    def test_kernel_imports_only_standard_library_and_itself(self):
        allowed = set(sys.stdlib_module_names) | {"delta_kernel", "__future__"}
        bad = [
            (str(path.relative_to(ROOT)), name, line)
            for path in kernel_files()
            for name, line in top_level_imports(path)
            if name not in allowed
        ]
        self.assertEqual(bad, [], f"Ядро импортирует не из стандартной библиотеки: {bad}")

    def test_kernel_has_no_float(self):
        bad = []
        for path in kernel_files():
            for node in ast.walk(parse(path)):
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    bad.append((str(path.relative_to(ROOT)), node.lineno, "литерал float"))
                elif isinstance(node, ast.Name) and node.id == "float":
                    bad.append((str(path.relative_to(ROOT)), node.lineno, "имя float"))
        self.assertEqual(bad, [], f"В ядре запрещён float: {bad}")


if __name__ == "__main__":
    unittest.main()
