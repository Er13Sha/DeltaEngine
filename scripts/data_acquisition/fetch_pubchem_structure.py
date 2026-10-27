"""Числовые данные о строении из PubChem PUG-REST.

- Периодическая таблица: электронная конфигурация, электроотрицательность (Полинг),
  энергия ионизации, сродство к электрону, радиус атома, степени окисления,
  температуры плавления и кипения, плотность.
- Структуры соединений базы: SMILES (граф связей атомов) и молекулярная формула.

    python scripts/data_acquisition/fetch_pubchem_structure.py data/sources/pubchem
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def get(url: str, attempts: int = 5) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "DeltaEngine research (PUG-REST, <5 req/s)"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_raw = get(f"{BASE}/periodictable/JSON")
    (out_dir / "periodic-table.json").write_bytes(table_raw)

    concepts = json.loads((ROOT / "data" / "confirmed-concepts.json").read_text(encoding="utf-8"))
    cids = [c["cid"] for c in concepts["compounds"]]
    rows = []
    for i in range(0, len(cids), 50):
        chunk = ",".join(str(c) for c in cids[i : i + 50])
        data = json.loads(get(f"{BASE}/compound/cid/{chunk}/property/MolecularFormula,SMILES,ConnectivitySMILES/JSON"))
        rows += data["PropertyTable"]["Properties"]
        time.sleep(0.3)
    structures = {
        "format": "pubchem-structures@0.1",
        "source": f"{BASE}/compound/cid/<cid>/property/MolecularFormula,SMILES,ConnectivitySMILES/JSON",
        "fetched": date.today().isoformat(),
        "compounds": rows,
    }
    (out_dir / "compound-structures.json").write_text(
        json.dumps(structures, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "format": "source-manifest@0.1",
        "source": "PubChem PUG-REST (NCBI), данные открытые",
        "fetched": date.today().isoformat(),
        "files": {
            "periodic-table.json": {"url": f"{BASE}/periodictable/JSON",
                                    "sha256": hashlib.sha256(table_raw).hexdigest()},
            "compound-structures.json": {"url": structures["source"], "compounds": len(rows)},
        },
        "status": "verified: значения скопированы из PubChem без изменений",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"таблица: {len(table_raw)} байт, структуры: {len(rows)} соединений", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
