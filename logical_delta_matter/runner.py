from __future__ import annotations

from pathlib import Path
from typing import Any

from .aggregate import classify_signature, compute_signature, detect_clusters
from .delta_core import Predicate, classify_delta, compile_delta_tree, validate_delta_tree
from .storage import append_event_log, read_json, save_checkpoint_json, write_json
from .templates import promote_all
from .viz import render_graph_svg, render_slice_svg
from .world_exact import WorldConfig, WorldState, config_from_dict, init_world, state_hash, state_to_dict, step_world_exact


DEFAULT_DELTA_TREE = {
    "name": "matter",
    "predicate": "is_discrete",
    "yes": {
        "name": "particles",
        "predicate": "is_grouped",
        "yes": {
            "name": "groups",
            "predicate": "is_stable",
            "yes": {"name": "substance"},
            "no": {"name": "transient_group"},
        },
        "no": {"name": "free_particles"},
    },
    "no": {"name": "field"},
}


PREDICATES: dict[str, Predicate] = {
    "is_discrete": lambda obj: bool(obj["is_discrete"]),
    "is_grouped": lambda obj: bool(obj["is_grouped"]),
    "is_stable": lambda obj: bool(obj["is_stable"]),
    "is_compact": lambda obj: bool(obj["is_compact"]),
}


def load_config(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def run_simulation(config_path: str | Path, out_dir: str | Path, export_ui_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    cfg = config_from_dict(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_log_path = out_dir / "events.jsonl"
    if event_log_path.exists():
        event_log_path.unlink()

    delta_tree = compile_delta_tree(config.get("delta_tree", DEFAULT_DELTA_TREE))
    validate_delta_tree(delta_tree, PREDICATES)

    state = init_world(cfg)
    latest = evaluate_state(state, cfg, delta_tree)
    write_outputs(state, latest, cfg, out_dir, config, export_ui_path)
    append_event_log(event_log_path, latest["event"])

    frame_every = int(config.get("visualization", {}).get("frame_every", 0))
    for _ in range(cfg.max_steps):
        state = step_world_exact(state, cfg)
        latest = evaluate_state(state, cfg, delta_tree)
        append_event_log(event_log_path, latest["event"])

        if frame_every and state.step % frame_every == 0:
            render_slice_svg(state, out_dir / "frames" / f"slice_{state.step:05d}.svg")

        if state.step % cfg.checkpoint_every == 0:
            save_checkpoint_json(
                out_dir / "checkpoints" / f"checkpoint_{state.step:05d}.json",
                state,
                latest["metrics"],
            )

    write_outputs(state, latest, cfg, out_dir, config, export_ui_path)
    return latest["summary"]


def evaluate_state(state: WorldState, cfg: WorldConfig, delta_tree: Any) -> dict[str, Any]:
    cluster_rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    signatures: list[dict[str, Any]] = []

    for cluster in detect_clusters(state):
        signature = compute_signature(cluster, state)
        signature_flags = classify_signature(signature)
        path = classify_delta(delta_tree, signature_flags, PREDICATES)
        leaf = path[-1]
        class_counts[leaf] = class_counts.get(leaf, 0) + 1
        signatures.append(signature)
        cluster_rows.append(
            {
                "id": cluster["id"],
                "size": cluster["size"],
                "signature": signature,
                "delta_path": path,
            }
        )

    promoted = promote_all(signatures)
    promoted_counts: dict[str, int] = {}
    for item in promoted:
        label = str(item["label"])
        promoted_counts[label] = promoted_counts.get(label, 0) + 1

    cluster_rows.sort(key=lambda row: (-int(row["size"]), int(row["id"])))
    metrics = {
        "step": state.step,
        "state_hash": state_hash(state),
        "n_particles": len(state.positions),
        "n_clusters": len(cluster_rows),
        "n_promoted": sum(1 for item in promoted if item["level"] != "cluster"),
        "occupancy_permille": (len(state.positions) * 1000) // cfg.volume,
        "logical_classes": dict(sorted(class_counts.items())),
        "promoted_counts": dict(sorted(promoted_counts.items())),
    }
    event = {
        "step": state.step,
        "state_hash": metrics["state_hash"],
        "n_clusters": metrics["n_clusters"],
        "n_promoted": metrics["n_promoted"],
        "logical_classes": metrics["logical_classes"],
    }
    summary = {
        "schema_version": "exact-mvp@0.1",
        "metrics": metrics,
        "top_clusters": cluster_rows[:12],
        "promoted": promoted[:12],
    }
    return {
        "metrics": metrics,
        "event": event,
        "summary": summary,
    }


def write_outputs(
    state: WorldState,
    latest: dict[str, Any],
    cfg: WorldConfig,
    out_dir: Path,
    config: dict[str, Any],
    export_ui_path: str | Path | None,
) -> None:
    summary = {
        **latest["summary"],
        "config": {
            "grid_shape": list(cfg.grid_shape),
            "max_steps": cfg.max_steps,
            "seed": cfg.seed,
            "neighborhood_radius": cfg.neighborhood_radius,
            "checkpoint_every": cfg.checkpoint_every,
            "particle_counts": cfg.particle_counts,
            "interaction_weights": cfg.interaction_weights,
        },
        "state": {
            key: value
            for key, value in state_to_dict(state).items()
            if key not in {"positions", "types"}
        },
        "source_config": config.get("name", "exact_mvp"),
    }
    write_json(out_dir / "summary.json", summary)
    render_slice_svg(state, out_dir / "latest_slice.svg")
    render_graph_svg(latest["summary"]["promoted"], out_dir / "latest_graph.svg")

    if export_ui_path:
        write_json(export_ui_path, summary)
