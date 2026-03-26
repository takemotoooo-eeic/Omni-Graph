from __future__ import annotations

from collections import defaultdict
import json

import streamlit as st
import streamlit.components.v1 as components

from services.session_state import get_rag_service, init_state


def _build_vis_network_html(nodes: list[dict], edges: list[dict], height: int = 640) -> str:
    vis_nodes = []
    for node in nodes:
        node_id = str(node["id"])
        vis_nodes.append(
            {
                "id": node_id,
                "label": str(node.get("label", node_id)),
                "title": f"id: {node_id}\ntype: {node.get('type', 'entity')}",
            }
        )

    vis_edges = []
    for edge in edges:
        vis_edges.append(
            {
                "from": str(edge["source"]),
                "to": str(edge["target"]),
                "label": str(edge.get("relation", "related_to")),
                "arrows": "to",
                "smooth": {"enabled": True, "type": "dynamic"},
            }
        )

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
      #kg {{
        width: 100%;
        height: {height}px;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        background: #FFFFFF;
      }}
    </style>
  </head>
  <body>
    <div id="kg"></div>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const container = document.getElementById("kg");
      const data = {{ nodes, edges }};
      const options = {{
        autoResize: true,
        interaction: {{
          dragNodes: true,
          dragView: true,
          zoomView: true,
          hover: true,
          navigationButtons: true,
          keyboard: true
        }},
        physics: {{
          enabled: true,
          stabilization: {{ iterations: 200 }}
        }},
        nodes: {{
          shape: "dot",
          size: 14,
          font: {{ size: 14 }},
          borderWidth: 1
        }},
        edges: {{
          font: {{ align: "middle", size: 10 }},
          color: {{ inherit: false, color: "#94A3B8" }}
        }}
      }};
      new vis.Network(container, data, options);
    </script>
  </body>
</html>
"""


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
        st.caption("マウスホイールでズーム、ドラッグで移動、ノードのドラッグで配置調整できます。")
        components.html(_build_vis_network_html(nodes, edges), height=680, scrolling=False)
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
