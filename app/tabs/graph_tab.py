from __future__ import annotations

from collections import defaultdict
import json

import streamlit as st
import streamlit.components.v1 as components

from services.session_state import get_rag_service, init_state


def _build_vis_network_html(
    nodes: list[dict],
    edges: list[dict],
    height: int = 640,
    show_node_labels: bool = True,
    show_edge_labels: bool = True,
) -> str:
    def _truncate(text: str, max_len: int = 800) -> str:
        text = str(text or "")
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _source_type_category(attributes: dict) -> str:
        """
        source_types に基づいて色カテゴリを決める。
        - text / image / table / equation が1種類だけ -> その種類
        - 2種類以上 -> 複数
        - 認識できる種類が無いが page_number がある -> text 扱い
        """
        raw = attributes.get("source_types") or ""
        tokens = [t.strip().lower() for t in str(raw).split(",") if t.strip()]
        recognized = {"text", "image", "table", "equation"}
        present = {t for t in tokens if t in recognized}
        if not present:
            if "page_number" in tokens:
                return "text"
            return "text"
        if len(present) == 1:
            return next(iter(present))
        return "複数"

    def _node_color(node: dict) -> dict:
        category = _source_type_category(node.get("attributes") or {})
        palette = {
            "text": "#60A5FA",  # blue
            "image": "#F59E0B",  # amber
            "table": "#10B981",  # green
            "equation": "#EF4444",  # red
            "複数": "#60A5FA",  # blue
        }
        background = palette.get(category, "#94A3B8")
        return {
            "background": background,
            "border": background,
            "highlight": {
                "background": background,
                "border": background,
            },
        }

    def _node_title(node: dict) -> str:
        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", "entity") or "entity")
        attributes = node.get("attributes") or {}

        entity_type = attributes.get("entity_type")
        primary_source_type = attributes.get("primary_source_type")
        source_types = attributes.get("source_types")
        source_id = attributes.get("source_id")
        file_path = attributes.get("file_path")
        created_at = attributes.get("created_at")
        truncate = attributes.get("truncate")
        description = attributes.get("description")
        keywords = attributes.get("keywords")

        lines: list[str] = [f"id: {node_id}", f"type: {node_type}"]
        if entity_type and str(entity_type) != node_type:
            lines.append(f"entity_type: {entity_type}")
        if primary_source_type:
            lines.append(f"primary_source_type: {primary_source_type}")
        if source_types:
            lines.append(f"source_types: {_truncate(source_types)}")
        if source_id:
            lines.append(f"source_id: {source_id}")
        if file_path:
            lines.append(f"file_path: {file_path}")
        if created_at:
            lines.append(f"created_at: {created_at}")
        if keywords:
            lines.append(f"keywords: {keywords}")
        if truncate:
            lines.append(f"truncate: {_truncate(truncate)}")
        if description:
            lines.append(f"description: {_truncate(description)}")
        return "\n".join(lines)

    def _edge_title(edge: dict) -> str:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        relation = str(edge.get("relation", "related_to") or "related_to")
        attributes = edge.get("attributes") or {}

        weight = attributes.get("weight")
        source_id = attributes.get("source_id")
        file_path = attributes.get("file_path")
        created_at = attributes.get("created_at")
        truncate = attributes.get("truncate")
        description = attributes.get("description")
        keywords = attributes.get("keywords")

        lines: list[str] = [f"{source} -> {target}", f"relation: {relation}"]
        if weight is not None and weight != "":
            lines.append(f"weight: {weight}")
        if source_id:
            lines.append(f"source_id: {source_id}")
        if file_path:
            lines.append(f"file_path: {file_path}")
        if created_at:
            lines.append(f"created_at: {created_at}")
        if keywords:
            lines.append(f"keywords: {keywords}")
        if truncate:
            lines.append(f"truncate: {_truncate(truncate)}")
        if description:
            lines.append(f"description: {_truncate(description)}")
        return "\n".join(lines)

    vis_nodes = []
    for node in nodes:
        node_id = str(node["id"])
        vis_nodes.append(
            {
                "id": node_id,
                "label": str(node.get("label", node_id)) if show_node_labels else "",
                "title": _node_title(node),
                "color": _node_color(node),
            }
        )

    vis_edges = []
    for edge in edges:
        vis_edges.append(
            {
                "from": str(edge["source"]),
                "to": str(edge["target"]),
                "label": str(edge.get("relation", "related_to")) if show_edge_labels else "",
                "arrows": "to",
                "smooth": {"enabled": True, "type": "dynamic"},
                "title": _edge_title(edge),
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

      /* vis-network のツールチップ幅を狭く固定 */
      .vis-tooltip {{
        width: 280px !important;
        max-width: 280px !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
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
        show_node_labels = st.checkbox("ノード名を表示", value=True)
        show_edge_labels = st.checkbox("エッジ名を表示", value=True)
        st.caption("マウスホイールでズーム、ドラッグで移動、ノードのドラッグで配置調整できます。")
        components.html(
            _build_vis_network_html(
                nodes,
                edges,
                show_node_labels=show_node_labels,
                show_edge_labels=show_edge_labels,
            ),
            height=680,
            scrolling=False,
        )
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
            st.markdown("#### 基本情報")
            st.json(
                {
                    "id": node.get("id"),
                    "label": node.get("label"),
                    "type": node.get("type"),
                }
            )
            st.markdown("#### 詳細情報")
            st.json(node.get("attributes", {}))
