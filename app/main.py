import streamlit as st

from app.tabs.chat_tab import render_chat_tab
from app.tabs.document_tab import render_document_tab
from app.tabs.graph_tab import render_graph_tab


def _init_page() -> None:
    st.set_page_config(
        page_title="RAGAnything Knowledge Graph App",
        page_icon="🧠",
        layout="wide",
    )
    st.title("RAGAnything Knowledge Graph App")
    st.caption("ドキュメント取り込み -> 知識グラフ可視化 -> グラフQAチャット")


def main() -> None:
    _init_page()

    tab_doc, tab_graph, tab_chat = st.tabs(
        ["ドキュメント入力", "グラフ可視化", "チャット"]
    )

    with tab_doc:
        render_document_tab()
    with tab_graph:
        render_graph_tab()
    with tab_chat:
        render_chat_tab()


if __name__ == "__main__":
    main()
