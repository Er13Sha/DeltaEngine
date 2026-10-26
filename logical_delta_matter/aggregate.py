from __future__ import annotations

from collections import Counter, deque
from typing import Any

from .world_exact import Cell, WorldState, neighbors


def detect_clusters(state: WorldState, min_size: int = 1) -> list[dict[str, Any]]:
    """Find 6-neighborhood connected components over occupied cells."""

    visited: set[Cell] = set()
    clusters: list[dict[str, Any]] = []

    for idx, start in enumerate(state.positions):
        if start in visited:
            continue

        queue: deque[Cell] = deque([start])
        visited.add(start)
        members: list[int] = []
        cells: list[Cell] = []

        while queue:
            cell = queue.popleft()
            member_idx = state.occupied[cell]
            members.append(member_idx)
            cells.append(cell)

            for adjacent in neighbors(cell, inferred_shape(state), include_self=False):
                if adjacent in state.occupied and adjacent not in visited:
                    visited.add(adjacent)
                    queue.append(adjacent)

        if len(members) >= min_size:
            clusters.append(
                {
                    "id": len(clusters),
                    "members": sorted(members),
                    "cells": sorted(cells),
                    "size": len(members),
                }
            )

    return clusters


def inferred_shape(state: WorldState) -> Cell:
    maxs = [0, 0, 0]
    for cell in state.positions:
        for axis in range(3):
            maxs[axis] = max(maxs[axis], cell[axis])
    return (maxs[0] + 2, maxs[1] + 2, maxs[2] + 2)


def compute_signature(cluster: dict[str, Any], state: WorldState) -> dict[str, Any]:
    cells: list[Cell] = cluster["cells"]
    xs = [cell[0] for cell in cells]
    ys = [cell[1] for cell in cells]
    zs = [cell[2] for cell in cells]
    bbox = (
        max(xs) - min(xs) + 1,
        max(ys) - min(ys) + 1,
        max(zs) - min(zs) + 1,
    )
    bbox_volume = bbox[0] * bbox[1] * bbox[2]
    type_counts = Counter(state.types[idx] for idx in cluster["members"])
    exposed_faces = 0
    cell_set = set(cells)

    for cell in cells:
        for adjacent in neighbors(cell, inferred_shape(state), include_self=False):
            if adjacent not in cell_set:
                exposed_faces += 1

    size = int(cluster["size"])
    return {
        "size": size,
        "bbox": list(bbox),
        "bbox_volume": int(bbox_volume),
        "surface": int(exposed_faces),
        "compactness_permille": int((size * 1000) // bbox_volume) if bbox_volume else 0,
        "type_counts": dict(sorted(type_counts.items())),
    }


def classify_signature(signature: dict[str, Any]) -> dict[str, bool]:
    size = int(signature["size"])
    return {
        "is_discrete": True,
        "is_grouped": size >= 2,
        "is_stable": size >= 3,
        "is_compact": int(signature["compactness_permille"]) >= 250,
    }


def summarize_clusters(state: WorldState, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in detect_clusters(state):
        signature = compute_signature(cluster, state)
        rows.append(
            {
                "id": cluster["id"],
                "size": cluster["size"],
                "signature": signature,
            }
        )
    rows.sort(key=lambda row: (-int(row["size"]), int(row["id"])))
    return rows[:limit]
