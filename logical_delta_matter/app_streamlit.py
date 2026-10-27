from __future__ import annotations

from pathlib import Path

from .runner import run_simulation


def run_streamlit_app() -> None:
    """Minimal Streamlit entry point.

    Streamlit is optional in the current workspace. This function imports it
    lazily so the exact core remains runnable without extra dependencies.
    """

    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Streamlit is not installed. Use the CLI in this workspace.") from exc

    st.set_page_config(page_title="Logical Delta Exact MVP", layout="wide")
    st.title("Logical Delta Exact MVP")
    config_path = st.text_input("Config", "configs/exact_mvp.json")
    out_dir = st.text_input("Output directory", "runs/streamlit_exact")

    if st.button("Run exact simulation"):
        summary = run_simulation(Path(config_path), Path(out_dir), Path("data/exact-latest.json"))
        st.json(summary["metrics"])
        slice_path = Path(out_dir) / "latest_slice.svg"
        if slice_path.exists():
            st.image(str(slice_path))
