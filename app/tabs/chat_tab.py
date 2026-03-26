from __future__ import annotations

import streamlit as st

from app.services.session_state import get_rag_service, init_state


def render_chat_tab() -> None:
    init_state()
    rag_service = get_rag_service()

    st.subheader("チャット")
    st.write("知識グラフを参照して質問応答を行います。")

    graph_data = st.session_state.get("graph_data", {"nodes": [], "edges": []})
    if not graph_data.get("nodes") and not graph_data.get("edges"):
        st.info("グラフがまだありません。先に「ドキュメント入力」タブでグラフを構築してください。")
        return

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("質問を入力してください")
    if not user_input:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("回答生成中..."):
            try:
                answer = rag_service.query(user_input, mode="hybrid")
            except Exception as exc:
                answer = f"回答生成に失敗しました: {exc}"
            st.markdown(answer)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
