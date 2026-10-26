from __future__ import annotations

from pathlib import Path
from typing import Any

from .world_exact import WorldState

TYPE_COLORS = {
    "proto_A": "#1f7a68",
    "proto_B": "#c46a2b",
    "proto_C": "#445d9c",
    "proto_D": "#9a5b10",
}


def render_slice_svg(state: WorldState, path: str | Path, z: int | None = None, cell_size: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not state.positions:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" />", encoding="utf-8")
        return

    max_x = max(cell[0] for cell in state.positions) + 1
    max_y = max(cell[1] for cell in state.positions) + 1
    max_z = max(cell[2] for cell in state.positions) + 1
    z = max_z // 2 if z is None else z
    width = max(1, max_x) * cell_size
    height = max(1, max_y) * cell_size

    rects: list[str] = []
    for idx, (x, y, pz) in enumerate(state.positions):
        if pz != z:
            continue
        color = TYPE_COLORS.get(state.types[idx], "#687174")
        rects.append(
            f'<rect x="{x * cell_size}" y="{y * cell_size}" width="{cell_size - 1}" '
            f'height="{cell_size - 1}" fill="{color}" />'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#fbfcf8" />'
        + "".join(rects)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def render_graph_svg(objects: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 720
    row_h = 44
    height = max(80, 24 + len(objects) * row_h)
    rows: list[str] = []
    for index, obj in enumerate(objects):
        y = 20 + index * row_h
        label = f"{obj.get('label', obj.get('level', 'object'))} / size={obj.get('signature', {}).get('size', '?')}"
        rows.append(f'<rect x="16" y="{y}" width="688" height="30" rx="6" fill="#f1faf6" stroke="#d8ddd4" />')
        rows.append(f'<text x="28" y="{y + 20}" font-size="13" font-family="Arial" fill="#1f2528">{escape_xml(label)}</text>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="#ffffff" />'
        + "".join(rows)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
