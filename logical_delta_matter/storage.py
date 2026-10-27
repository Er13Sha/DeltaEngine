from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .world_exact import WorldState, state_from_dict, state_to_dict


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_event_log(path: str | Path, event: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def save_checkpoint_json(path: str | Path, state: WorldState, meta: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "state": state_to_dict(state),
            "meta": meta,
        },
    )


def load_checkpoint_json(path: str | Path) -> tuple[WorldState, dict[str, Any]]:
    data = read_json(path)
    return state_from_dict(data["state"]), data.get("meta", {})


def save_checkpoint_hdf5(path: str | Path, state: WorldState, meta: dict[str, Any]) -> None:
    """Compatibility hook.

    The current project does not require h5py. If h5py is installed later, this
    function can be replaced with chunked HDF5 storage without changing callers.
    For now it writes a JSON checkpoint next to the requested path.
    """

    json_path = Path(path).with_suffix(".json")
    save_checkpoint_json(json_path, state, {**meta, "storage_note": "HDF5 fallback: JSON checkpoint"})
