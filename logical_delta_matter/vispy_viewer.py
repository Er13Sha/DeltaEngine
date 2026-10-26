from __future__ import annotations

from typing import Sequence


def run_vispy_demo(
    positions: Sequence[tuple[float, float, float]] | None = None,
    radial_curve: Sequence[tuple[float, float]] | None = None,
) -> dict[str, str] | None:
    """Launch a minimal VisPy 3D particle and 2D velocity-curve viewer."""

    try:
        from vispy import app, scene
    except ModuleNotFoundError:
        return {
            "status": "missing_dependency",
            "dependency": "vispy",
            "message": "Install vispy to run the OpenGL 3D/2D viewer.",
        }

    positions = positions or [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.5, 0.0)]
    radial_curve = radial_curve or [(0.5, 0.2), (1.0, 0.7), (2.0, 1.1), (3.0, 1.2)]

    canvas = scene.SceneCanvas(keys="interactive", show=True, title="Logical Delta Matter VisPy")
    grid = canvas.central_widget.add_grid()

    view3d = grid.add_view(row=0, col=0, border_color="white")
    view3d.camera = "turntable"
    markers = scene.visuals.Markers(parent=view3d.scene)
    markers.set_data(positions, face_color=(0.2, 0.55, 0.9, 1.0), size=8)

    view2d = grid.add_view(row=1, col=0, border_color="white")
    view2d.camera = "panzoom"
    line = scene.visuals.Line(
        pos=[(radius, velocity) for radius, velocity in radial_curve],
        color=(0.1, 0.8, 0.45, 1.0),
        width=2,
        parent=view2d.scene,
    )
    _ = line
    app.run()
    return None


if __name__ == "__main__":
    result = run_vispy_demo()
    if result:
        print(result["message"])
