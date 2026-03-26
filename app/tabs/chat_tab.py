from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from services.session_state import get_rag_service, init_state


def _truncate(text: Any, max_len: int = 600) -> str:
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _source_type_category_from_attributes(attributes: dict[str, Any]) -> str:
    raw = attributes.get("source_types") or ""
    tokens = [t.strip().lower() for t in str(raw).split(",") if t.strip()]
    recognized = {"text", "image", "table", "equation"}
    present = {t for t in tokens if t in recognized}
    if not present:
        # page_number のみ等は text 扱い（既存仕様に合わせる）
        if "page_number" in tokens:
            return "text"
        return "text"
    if len(present) == 1:
        return next(iter(present))
    return "複数"


def _node_color_from_attributes(attributes: dict[str, Any], emphasized: bool) -> dict[str, Any]:
    if not emphasized:
        grey = "#CBD5E1"
        return {"background": grey, "border": grey, "highlight": {"background": grey, "border": grey}}

    # チャットタブでは「取得されたノードだけ目立たせる」ため、
    # タイプによる色分けはせずオレンジに統一する。
    background = "#F59E0B"
    return {
        "background": background,
        "border": background,
        "highlight": {"background": background, "border": background},
    }


def _edge_color(emphasized: bool) -> dict[str, Any]:
    if emphasized:
        color = "#F59E0B"
        return {"color": color, "highlight": color}
    return {"color": "#CBD5E1", "highlight": "#CBD5E1"}


def _build_context_vis_network_html(
    full_nodes: list[dict[str, Any]],
    full_edges: list[dict[str, Any]],
    emphasized_node_ids: set[str],
    emphasized_edge_pairs: set[tuple[str, str]],
    height: int = 520,
) -> str:
    # vis-network へ渡すため、ノード/エッジを加工
    vis_nodes: list[dict[str, Any]] = []
    for node in full_nodes:
        node_id = str(node.get("id", ""))
        attrs = node.get("attributes") or {}
        emphasized = node_id in emphasized_node_ids

        label = str(node.get("label", node_id)) if emphasized else ""
        description = attrs.get("description") or ""
        file_path = attrs.get("file_path") or ""
        entity_type = attrs.get("entity_type") or attrs.get("type") or ""

        title_lines = [
            f"id: {node_id}",
            f"type: {entity_type or node.get('type', 'entity')}",
        ]
        if file_path:
            title_lines.append(f"file_path: {file_path}")
        if description:
            title_lines.append(f"description: {_truncate(description)}")

        vis_nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": "\n".join(title_lines),
                "color": _node_color_from_attributes(attrs, emphasized),
            }
        )

    vis_edges: list[dict[str, Any]] = []
    for edge in full_edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        # undirected 対応：source/target の順が違っても同一ペア扱い
        pair = tuple(sorted([source, target]))
        emphasized = pair in emphasized_edge_pairs

        attrs = edge.get("attributes") or {}
        description = attrs.get("description") or ""
        file_path = attrs.get("file_path") or ""
        weight = attrs.get("weight")

        title_lines = [f"{source} -> {target}"]
        if weight is not None and weight != "":
            title_lines.append(f"weight: {weight}")
        if file_path:
            title_lines.append(f"file_path: {file_path}")
        if description:
            title_lines.append(f"description: {_truncate(description)}")

        vis_edges.append(
            {
                "from": source,
                "to": target,
                "label": "",  # チャット表示ではラベルは抑制（色と太さで強調）
                "arrows": "to",
                "smooth": {"enabled": True, "type": "dynamic"},
                "title": "\n".join(title_lines),
                "color": _edge_color(emphasized),
                "width": 3 if emphasized else 1,
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
    <div id="kg" style="width: 100%; height: {height}px; border: 1px solid #E5E7EB; border-radius: 10px; background: #FFFFFF;"></div>
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
          stabilization: {{ iterations: 120 }}
        }},
        nodes: {{
          shape: "dot",
          size: 14,
          font: {{ size: 12 }},
          borderWidth: 1
        }},
        edges: {{
          smooth: true
        }}
      }};
      new vis.Network(container, data, options);
    </script>
  </body>
</html>
"""


def _render_image_grid(
    images: list[Any],
    *,
    fixed_height: int = 220,
    max_show: int = 10,
    columns_per_row: int = 4,
) -> None:
    # st.image は height 引数を受け付けないため、CSS で疑似的に高さ固定する
    st.markdown(
        f"""
        <style>
          /* Streamlit の st.image は data-testid="stImage" の中に img が入る */
          div[data-testid="stImage"] img {{
            height: {fixed_height}px !important;
            width: 100% !important;
            object-fit: contain !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    valid_paths = [p for p in images[:max_show] if isinstance(p, str) and p]
    if not valid_paths:
        return

    # 省略分がある場合はキャプションを最後に出す
    omitted = len([p for p in images if isinstance(p, str) and p]) - len(valid_paths)
    for start in range(0, len(valid_paths), columns_per_row):
        row = valid_paths[start : start + columns_per_row]
        cols = st.columns(len(row))
        for col, img_path in zip(cols, row):
            p = Path(img_path)
            if not p.exists():
                col.warning(f"画像が見つかりません: {img_path}")
                continue
            # 高さは CSS 側で固定（上の style）
            col.image(img_path)

    if omitted > 0:
        st.caption(f"（{omitted} 件は省略）")


def render_chat_tab() -> None:
    init_state()
    rag_service = get_rag_service()

    st.subheader("チャット")
    st.write("知識グラフを参照して質問応答を行います。")

    show_context_ui = st.checkbox("コンテキスト（画像/グラフ）を表示", value=True)

    graph_data = st.session_state.get("graph_data", {"nodes": [], "edges": []})
    if not graph_data.get("nodes") and not graph_data.get("edges"):
        st.info("グラフがまだありません。先に「ドキュメント入力」タブでグラフを構築してください。")
        return

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if show_context_ui and msg.get("images"):
                st.markdown("#### 取得画像")
                _render_image_grid(msg.get("images") or [], fixed_height=220, max_show=10)

            if show_context_ui and msg.get("context"):
                st.markdown("#### 参照グラフ")
                ctx = msg.get("context") or {}
                ctx_nodes = ctx.get("nodes") or []
                ctx_edges = ctx.get("edges") or []
                emphasized_node_ids = {str(n.get("id")) for n in ctx_nodes if isinstance(n, dict) and n.get("id")}
                emphasized_edge_pairs: set[tuple[str, str]] = set()
                for e in ctx_edges:
                    if not isinstance(e, dict):
                        continue
                    s = e.get("source")
                    t = e.get("target")
                    if s and t:
                        emphasized_edge_pairs.add(tuple(sorted([str(s), str(t)])))

                if emphasized_node_ids:
                    components.html(
                        _build_context_vis_network_html(
                            full_nodes=graph_data.get("nodes", []),
                            full_edges=graph_data.get("edges", []),
                            emphasized_node_ids=emphasized_node_ids,
                            emphasized_edge_pairs=emphasized_edge_pairs,
                            height=520,
                        ),
                        height=560,
                        scrolling=False,
                    )
                else:
                    st.caption("取得されたノードはありませんでした。")

    user_input = st.chat_input("質問を入力してください")
    if not user_input:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("回答生成中..."):
            try:
                if show_context_ui:
                    ctx_result = rag_service.query_with_context(
                        user_input, mode="hybrid"
                    )
                    answer = ctx_result.get("return", "")
                    images = ctx_result.get("images") or []
                    ctx_nodes = ctx_result.get("nodes") or []
                    ctx_edges = ctx_result.get("edges") or []
                    context_payload = {
                        "nodes": ctx_nodes,
                        "edges": ctx_edges,
                    }
                else:
                    answer = rag_service.query(user_input, mode="hybrid")
                    images = []
                    context_payload = None
            except Exception as exc:
                answer = f"回答生成に失敗しました: {exc}"
                images = []
                context_payload = None
            st.markdown(answer)

            if show_context_ui and context_payload is not None:
                if images:
                    st.markdown("#### 参照画像")
                    _render_image_grid(images, fixed_height=220, max_show=10)

                st.markdown("#### 参照グラフ")
                ctx_nodes = context_payload.get("nodes") or []
                ctx_edges = context_payload.get("edges") or []
                emphasized_node_ids = {
                    str(n.get("id"))
                    for n in ctx_nodes
                    if isinstance(n, dict) and n.get("id")
                }
                emphasized_edge_pairs: set[tuple[str, str]] = set()
                for e in ctx_edges:
                    if not isinstance(e, dict):
                        continue
                    s = e.get("source")
                    t = e.get("target")
                    if s and t:
                        emphasized_edge_pairs.add(tuple(sorted([str(s), str(t)])))

                if emphasized_node_ids:
                    components.html(
                        _build_context_vis_network_html(
                            full_nodes=graph_data.get("nodes", []),
                            full_edges=graph_data.get("edges", []),
                            emphasized_node_ids=emphasized_node_ids,
                            emphasized_edge_pairs=emphasized_edge_pairs,
                            height=520,
                        ),
                        height=560,
                        scrolling=False,
                    )
                else:
                    st.caption("取得されたノードはありませんでした。")

    assistant_msg: dict[str, Any] = {"role": "assistant", "content": answer}
    if show_context_ui:
        # 画像/コンテキストは message 内に保存して、リロード時も表示できるようにする
        assistant_msg["images"] = images if "images" in locals() else []
        assistant_msg["context"] = context_payload if "context_payload" in locals() else None
    st.session_state["chat_history"].append(assistant_msg)
