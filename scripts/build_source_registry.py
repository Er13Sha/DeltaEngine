"""Пересборка реестра источников data/source-registry.json (SHA-256).

Запуск из любого каталога: python scripts/build_source_registry.py

В реестр входят первичные источники и курируемые данные. Производные файлы
(data/computed-compound-space.json, data/exact-latest.json) и документы проекта не входят:
первые воспроизводятся командами из README.md, вторые меняются постоянно.

Отличие от версии в LogicalDeltaCorrect: отсутствующий источник — ошибка, а не молчаливый пропуск;
пути записываются с прямыми слэшами.

TODO (этап M0): расширить запись полями status и anchor (source-registry@0.2).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logical_delta_matter.source_registry import SourceRegistry
from logical_delta_matter.storage import write_json

FIXED_SOURCES = [
    ("source.deep_research_report", "docs/research/deep-research-report.md"),
    ("source.katyushchik_monograph", "data/sources/documents/monografia__katyuschik_v_g.pdf"),
    ("source.delta_ai_pdf", "data/sources/documents/Архитектура ИИ на логических дельтах.pdf"),
    ("source.galaxy_model_pdf", "data/sources/documents/Моделирование галактики по теории Катющика.pdf"),
    ("source.gravity_calculation_pdf", "data/sources/documents/Улучшение расчетов гравитационного взаимодействия.pdf"),
    ("data.matter_deltas", "data/matter-deltas.json"),
    ("data.rule_packs", "data/rule-packs.json"),
    ("data.confirmed_concepts", "data/confirmed-concepts.json"),
    ("data.compound_seed", "data/compound-seed.json"),
    ("data.compound_space_config", "data/compound-space-config.json"),
]

# (префикс идентификатора, каталог, шаблон файлов, как получить суффикс идентификатора)
GLOBBED_SOURCES = [
    ("source.video.", "data/sources/video-transcripts", "*.ru-auto.txt", lambda p: p.name.split(".")[0]),
    ("source.forum.", "data/sources/forum-topics", "t*.txt", lambda p: p.stem),
    ("source.image.", "data/sources/images", "*.jpg", lambda p: p.stem),
    ("source.manifest.", "data/sources", "*/manifest.json", lambda p: p.parent.name),
]


def main() -> int:
    os.chdir(ROOT)
    registry = SourceRegistry()

    for source_id, rel in FIXED_SOURCES:
        path = Path(rel)
        if not path.exists():
            raise SystemExit(f"Отсутствует источник: {rel}")
        registry.register_file(source_id, path)

    for prefix, directory, pattern, suffix in GLOBBED_SOURCES:
        for path in sorted(Path(directory).glob(pattern)):
            registry.register_file(prefix + suffix(path), path)

    manifest = registry.to_manifest()
    for record in manifest["records"]:
        record["path"] = record["path"].replace("\\", "/")
    write_json("data/source-registry.json", manifest)
    print(f"Записей в реестре: {len(manifest['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
