from __future__ import annotations

import streamlit as st

from app.services.session_state import get_rag_service, init_state
from app.utils.file_utils import save_uploaded_files


def render_document_tab() -> None:
    init_state()
    rag_service = get_rag_service()

    st.subheader("ドキュメント入力")
    st.write("ドキュメントをアップロードし、知識グラフを構築します。")

    uploaded_files = st.file_uploader(
        "ファイルを選択",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="対応形式: pdf, txt, md, docx",
    )

    can_build = bool(uploaded_files)
    if st.button("グラフ構築開始", disabled=not can_build, type="primary"):
        with st.spinner("ドキュメントを処理して知識グラフを構築中..."):
            try:
                saved_paths = save_uploaded_files(
                    upload_dir=f"{rag_service.working_dir}/uploads",
                    uploaded_files=uploaded_files or [],
                )
                result = rag_service.process_documents(saved_paths)
                graph_data = rag_service.get_graph_data()

                st.session_state["build_result"] = result
                st.session_state["graph_data"] = graph_data

            except Exception as exc:
                st.error(f"グラフ構築に失敗しました: {exc}")
                return

        st.success("グラフ構築が完了しました。")

    build_result = st.session_state.get("build_result")
    if build_result:
        st.markdown("### 処理サマリ")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("成功件数", build_result.get("processed", 0))
        col2.metric("失敗件数", len(build_result.get("failed", [])))
        col3.metric("ノード数", build_result.get("node_count", 0))
        col4.metric("エッジ数", build_result.get("edge_count", 0))

        failed = build_result.get("failed", [])
        if failed:
            st.warning("一部ファイルで失敗しました。")
            st.json(failed)
