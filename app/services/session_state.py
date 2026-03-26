from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from services.rag_service import RAGService


def get_rag_service() -> RAGService:
    if "rag_service" not in st.session_state:
        working_dir = os.getenv("APP_WORKING_DIR", "./rag_storage/streamlit_app")
        output_dir = os.getenv("APP_OUTPUT_DIR", "./output/streamlit_app")
        parser = os.getenv("PARSER", "mineru")

        Path(working_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        st.session_state["rag_service"] = RAGService(
            working_dir=working_dir,
            output_dir=output_dir,
            parser=parser,
        )
    return st.session_state["rag_service"]


def init_state() -> None:
    st.session_state.setdefault("graph_data", {"nodes": [], "edges": []})
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("build_result", None)
