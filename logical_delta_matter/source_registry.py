from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .storage import write_json


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SourceRegistry:
    def __init__(self) -> None:
        self._records: dict[str, SourceRecord] = {}

    def register_file(self, source_id: str, path: str | Path) -> SourceRecord:
        resolved = Path(path)
        stat = resolved.stat()
        record = SourceRecord(
            source_id=source_id,
            path=str(resolved),
            sha256=sha256_file(resolved),
            size_bytes=stat.st_size,
        )
        self._records[source_id] = record
        return record

    def verify(self, source_id: str) -> bool:
        record = self._records[source_id]
        path = Path(record.path)
        return path.exists() and path.stat().st_size == record.size_bytes and sha256_file(path) == record.sha256

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "source-registry@0.1",
            "hash_algorithm": "sha256",
            "records": [record.to_dict() for record in self._records.values()],
        }

    def write_manifest(self, path: str | Path) -> None:
        write_json(path, self.to_manifest())
