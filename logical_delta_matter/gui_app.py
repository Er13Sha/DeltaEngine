from __future__ import annotations

from .galaxy_shadow import ShadowConfig, direct_shadow_forces_cpu


def run_gui() -> dict[str, str] | None:
    """Launch a minimal DearPyGui control panel when optional GUI deps exist."""

    try:
        import dearpygui.dearpygui as dpg
    except ModuleNotFoundError:
        return {
            "status": "missing_dependency",
            "dependency": "dearpygui",
            "message": "Install dearpygui and vispy to run the interactive GUI.",
        }

    state = {"pressure": 1.0, "shadow_coeff": 1.0}

    def recompute() -> None:
        state["pressure"] = dpg.get_value("pressure")
        state["shadow_coeff"] = dpg.get_value("shadow_coeff")
        forces = direct_shadow_forces_cpu(
            [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            [1.0, 1.0],
            ShadowConfig(cosmic_pressure=state["pressure"], shadow_coeff=state["shadow_coeff"]),
        )
        dpg.set_value("force_text", f"F0={forces[0]} F1={forces[1]}")

    dpg.create_context()
    dpg.create_viewport(title="Logical Delta Matter GUI", width=980, height=720)
    with dpg.window(label="Галактическое давление и логические дельты", width=960, height=680):
        dpg.add_text("Research prototype: pressure-shadow direct O(N^2) interaction")
        dpg.add_slider_float(label="Давление Вселенной", tag="pressure", default_value=1.0, min_value=0.0, max_value=10.0, callback=recompute)
        dpg.add_slider_float(label="Коэффициент тени", tag="shadow_coeff", default_value=1.0, min_value=0.0, max_value=10.0, callback=recompute)
        dpg.add_text("", tag="force_text")
        dpg.add_button(label="Пересчитать", callback=recompute)
    recompute()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
    return None


if __name__ == "__main__":
    result = run_gui()
    if result:
        print(result["message"])
