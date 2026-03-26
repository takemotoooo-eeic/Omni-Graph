from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig


class RAGService:
    def __init__(self, working_dir: str, output_dir: str, parser: str) -> None:
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.parser = parser
        self.rag: RAGAnything | None = None

    def ensure_initialized(self) -> None:
        if self.rag is not None:
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("環境変数 GEMINI_API_KEY が設定されていません。")

        gemini_client = genai.Client(api_key=api_key)
        llm_model = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
        embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser=self.parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        def _history_to_text(history_messages: list[dict[str, Any]] | None) -> str:
            if not history_messages:
                return ""
            lines: list[str] = []
            for msg in history_messages:
                role = str(msg.get("role", "user")).upper()
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                parts.append(str(c.get("text", "")))
                            elif "text" in c:
                                parts.append(str(c["text"]))
                    content_text = "\n".join([p for p in parts if p])
                else:
                    content_text = str(content)
                lines.append(f"{role}: {content_text}")
            return "\n".join(lines)

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            merged_prompt = prompt
            history_text = _history_to_text(history_messages or [])
            if history_text:
                merged_prompt = (
                    "Conversation history:\n"
                    f"{history_text}\n\n"
                    f"Current user request:\n{prompt}"
                )

            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=llm_model,
                contents=merged_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None
                ),
            )
            return response.text or ""

        async def vision_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            messages: list[dict[str, Any]] | None = None,
            image_data: str | None = None,
            **kwargs: Any,
        ) -> str:
            if messages:
                multimodal_parts: list[Any] = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content")
                    if isinstance(content, str):
                        multimodal_parts.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if not isinstance(c, dict):
                                continue
                            if c.get("type") == "text":
                                multimodal_parts.append(str(c.get("text", "")))
                            elif c.get("type") == "image_url":
                                raw = c.get("image_url", {}).get("url", "").split(",")[-1]
                                if raw:
                                    image_bytes = base64.b64decode(raw)
                                    multimodal_parts.append(
                                        types.Part.from_bytes(
                                            data=image_bytes, mime_type="image/jpeg"
                                        )
                                    )

                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=llm_model,
                    contents=multimodal_parts if multimodal_parts else prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None
                    ),
                )
                return response.text or ""
            if image_data:
                image_bytes = base64.b64decode(image_data)
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=llm_model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None
                    ),
                )
                return response.text or ""
            return await llm_model_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        async def gemini_embed_func(texts: list[str]) -> np.ndarray:
            response = await asyncio.to_thread(
                gemini_client.models.embed_content,
                model=embedding_model,
                contents=texts,
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                raise ValueError("Gemini embedding response missing `embeddings` field.")

            vectors = [np.array(item.values, dtype=np.float32) for item in embeddings]
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Gemini embedding count mismatch: requested {len(texts)} texts but got {len(vectors)} vectors."
                )
            return np.array(vectors)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=gemini_embed_func,
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

    def process_documents(self, file_paths: list[str]) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.rag is not None

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        success = 0
        failed: list[dict[str, str]] = []

        for file_path in file_paths:
            try:
                asyncio.run(
                    self.rag.process_document_complete(
                        file_path=file_path,
                        output_dir=self.output_dir,
                        parse_method="auto",
                    )
                )
                success += 1
            except Exception as exc:
                failed.append({"file": file_path, "error": str(exc)})

        graph = self.get_graph_data()
        return {
            "processed": success,
            "failed": failed,
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
        }

    def query(self, text: str, mode: str = "hybrid") -> str:
        self._ensure_lightrag_ready()
        assert self.rag is not None
        return asyncio.run(self.rag.aquery(text, mode=mode))

    def _ensure_lightrag_ready(self) -> None:
        self.ensure_initialized()
        assert self.rag is not None

        initializer = getattr(self.rag, "_ensure_lightrag_initialized", None)
        if not callable(initializer):
            return

        result = asyncio.run(initializer())
        if isinstance(result, dict) and not result.get("success", True):
            raise RuntimeError(
                result.get("error", "LightRAG の初期化に失敗しました。")
            )

    def get_graph_data(self) -> dict[str, list[dict[str, Any]]]:
        self.ensure_initialized()
        assert self.rag is not None
        graph = self._try_get_graph_from_rag_instance()
        if graph["nodes"] or graph["edges"]:
            return graph
        return self._load_graph_from_files()

    def _try_get_graph_from_rag_instance(self) -> dict[str, list[dict[str, Any]]]:
        assert self.rag is not None
        lightrag = getattr(self.rag, "lightrag", None)
        if lightrag is None:
            return {"nodes": [], "edges": []}

        graph_obj = getattr(lightrag, "chunk_entity_relation_graph", None)
        if graph_obj is None:
            return {"nodes": [], "edges": []}

        method_names = [
            "to_dict",
            "get_graph",
            "dump",
            "get_all",
            "export",
        ]
        for method_name in method_names:
            method = getattr(graph_obj, method_name, None)
            if not callable(method):
                continue
            try:
                result = asyncio.run(method()) if inspect.iscoroutinefunction(method) else method()
                normalized = self._normalize_graph_payload(result)
                if normalized["nodes"] or normalized["edges"]:
                    return normalized
            except Exception:
                continue

        return {"nodes": [], "edges": []}

    def _load_graph_from_files(self) -> dict[str, list[dict[str, Any]]]:
        base = Path(self.working_dir)
        if not base.exists():
            return {"nodes": [], "edges": []}

        candidates = sorted(base.rglob("*"))
        for path in candidates:
            lower = path.name.lower()
            if not path.is_file():
                continue
            if ("graph" not in lower and "entity" not in lower and "relation" not in lower):
                continue

            if path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    normalized = self._normalize_graph_payload(data)
                    if normalized["nodes"] or normalized["edges"]:
                        return normalized
                except Exception:
                    continue

            if path.suffix.lower() == ".graphml":
                normalized = self._parse_graphml(path)
                if normalized["nodes"] or normalized["edges"]:
                    return normalized

        return {"nodes": [], "edges": []}

    def _normalize_graph_payload(self, data: Any) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(data, dict):
            return {"nodes": [], "edges": []}

        nodes = data.get("nodes") or data.get("vertices") or []
        edges = data.get("edges") or data.get("links") or data.get("relations") or []

        normalized_nodes: list[dict[str, Any]] = []
        for n in nodes if isinstance(nodes, list) else []:
            if not isinstance(n, dict):
                continue
            node_id = str(n.get("id") or n.get("node_id") or n.get("name") or n.get("label") or "")
            if not node_id:
                continue
            normalized_nodes.append(
                {
                    "id": node_id,
                    "label": str(n.get("label") or n.get("name") or node_id),
                    "type": str(n.get("type") or "entity"),
                    "attributes": n,
                }
            )

        normalized_edges: list[dict[str, Any]] = []
        for e in edges if isinstance(edges, list) else []:
            if not isinstance(e, dict):
                continue
            source = str(e.get("source") or e.get("from") or e.get("src") or "")
            target = str(e.get("target") or e.get("to") or e.get("dst") or "")
            if not source or not target:
                continue
            normalized_edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": str(e.get("relation") or e.get("label") or "related_to"),
                    "attributes": e,
                }
            )

        return {"nodes": normalized_nodes, "edges": normalized_edges}

    def _parse_graphml(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            return {"nodes": [], "edges": []}

        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for node in root.findall(".//g:node", ns):
            node_id = node.attrib.get("id", "")
            if not node_id:
                continue
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "type": "entity",
                    "attributes": dict(node.attrib),
                }
            )

        for edge in root.findall(".//g:edge", ns):
            source = edge.attrib.get("source", "")
            target = edge.attrib.get("target", "")
            if not source or not target:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "relation": edge.attrib.get("label", "related_to"),
                    "attributes": dict(edge.attrib),
                }
            )

        return {"nodes": nodes, "edges": edges}
