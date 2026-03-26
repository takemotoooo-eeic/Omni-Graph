from __future__ import annotations

from collections import defaultdict

import streamlit as st

from app.services.session_state import get_rag_service, init_state


def _to_dot(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["digraph G {", 'rankdir="LR";', 'node [shape=ellipse, style=filled, fillcolor="#E8F0FE"];']

    node_ids = {n["id"] for n in nodes}
    for n in nodes:
        safe_label = str(n.get("label", n["id"])).replace('"', "'")
        lines.append(f'"{n["id"]}" [label="{safe_label}"];')

    for e in edges:
        if e["source"] not in node_ids:
            lines.append(f'"{e["source"]}" [label="{e["source"]}"];')
        if e["target"] not in node_ids:
            lines.append(f'"{e["target"]}" [label="{e["target"]}"];')
        rel = str(e.get("relation", "related_to")).replace('"', "'")
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{rel}"];')

    lines.append("}")
    return "\n".join(lines)


def render_graph_tab() -> None:
    init_state()
    rag_service = get_rag_service()

    st.subheader("グラフ可視化")
    st.write("構築済みの知識グラフを可視化します。")

    if st.button("最新グラフを再読込"):
        try:
            st.session_state["graph_data"] = rag_service.get_graph_data()
            st.success("グラフを更新しました。")
        except Exception as exc:
            st.error(f"グラフ更新に失敗しました: {exc}")

    graph_data = st.session_state.get("graph_data", {"nodes": [], "edges": []})
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes and not edges:
        st.info("グラフがまだありません。先に「ドキュメント入力」タブで構築してください。")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.graphviz_chart(_to_dot(nodes, edges), use_container_width=True)
    with col2:
        st.metric("ノード数", len(nodes))
        st.metric("エッジ数", len(edges))

        relation_count: dict[str, int] = defaultdict(int)
        for e in edges:
            relation_count[str(e.get("relation", "related_to"))] += 1
        if relation_count:
            st.markdown("**関係タイプ内訳**")
            st.json(dict(relation_count))

    st.markdown("### ノード詳細")
    node_options = [n["id"] for n in nodes]
    selected = st.selectbox("ノードを選択", options=node_options, index=0 if node_options else None)
    if selected:
        node = next((n for n in nodes if n["id"] == selected), None)
        if node:
            st.json(node)
