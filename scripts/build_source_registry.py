"""Пересборка реестра источников data/source-registry.json (SHA-256).

Запуск из любого каталога: python scripts/build_source_registry.py

Только стандартная библиотека. В реестр входят первичные источники и курируемые данные;
документы проекта (docs/*.md) и производные файлы в него не входят. Отсутствующий источник —
ошибка, а не молчаливый пропуск. Пути записываются с прямыми слэшами.

TODO (этап M0): расширить запись полями status и anchor (source-registry@0.2).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIXED_SOURCES = [
    ("source.deep_research_report", "docs/research/deep-research-report.md"),
    ("source.katyushchik_monograph", "data/sources/documents/monografia__katyuschik_v_g.pdf"),
    ("source.delta_ai_pdf", "data/sources/documents/Архитектура ИИ на логических дельтах.pdf"),
    ("source.galaxy_model_pdf", "data/sources/documents/Моделирование галактики по теории Катющика.pdf"),
    ("source.gravity_calculation_pdf", "data/sources/documents/Улучшение расчетов гравитационного взаимодействия.pdf"),
    ("data.matter_deltas", "data/matter-deltas.json"),
    ("data.rule_packs", "data/rule-packs.json"),
    ("data.confirmed_concepts", "data/confirmed-concepts.json"),
]

# (префикс идентификатора, каталог, шаблон файлов, как получить суффикс идентификатора)
GLOBBED_SOURCES = [
    ("source.video.", "data/sources/video-transcripts", "*.ru-auto.txt", lambda p: p.name.split(".")[0]),
    ("source.forum.", "data/sources/forum-topics", "t*.txt", lambda p: p.stem),
    ("source.image.", "data/sources/images", "*.jpg", lambda p: p.stem),
    ("source.manifest.", "data/sources", "*/manifest.json", lambda p: p.parent.name),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    os.chdir(ROOT)
    records: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(source_id: str, path: Path) -> None:
        if source_id in seen:
            raise SystemExit(f"Повторяющийся идентификатор источника: {source_id}")
        seen.add(source_id)
        records.append({
            "source_id": source_id,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })

    for source_id, rel in FIXED_SOURCES:
        path = Path(rel)
        if not path.exists():
            raise SystemExit(f"Отсутствует источник: {rel}")
        add(source_id, path)

    for prefix, directory, pattern, suffix in GLOBBED_SOURCES:
        for path in sorted(Path(directory).glob(pattern)):
            add(prefix + suffix(path), path)

    manifest = {"schema_version": "source-registry@0.1", "hash_algorithm": "sha256", "records": records}
    Path("data/source-registry.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Записей в реестре: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
