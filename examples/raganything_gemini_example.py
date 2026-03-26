#!/usr/bin/env python
"""
Gemini API key example script demonstrating parser integration with RAGAnything.

このサンプルでは .env の GEMINI_API_KEY を使い、
- Embedding: gemini-embedding-2-preview
- LLM: gemini-3-flash-preview
を Google 公式ライブラリ経由で呼び出して RAGAnything を実行します。
"""

import os
import argparse
import asyncio
import base64
import logging
import logging.config
from pathlib import Path
from typing import Any

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application."""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "raganything_gemini_example.log")
    )

    print(f"\nRAGAnything Gemini example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    working_dir: str | None = None,
    parser: str = "mineru",
    llm_model: str = "gemini-3-flash-preview",
    embedding_model: str = "gemini-embedding-2-preview",
):
    """Process document with RAGAnything using native Gemini SDK."""
    try:
        gemini_client = genai.Client(api_key=api_key)

        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,
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
            prompt, system_prompt=None, history_messages=None, **kwargs
        ):
            if history_messages is None:
                history_messages = []

            merged_prompt = prompt
            history_text = _history_to_text(history_messages)
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
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if history_messages is None:
                history_messages = []

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
                                raw = (
                                    c.get("image_url", {}).get("url", "").split(",")[-1]
                                )
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
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # gemini-embedding-2-preview currently returns 3072-dim vectors.
        # LightRAG validates vector count from (total_elements / embedding_dim),
        # so this value must match the actual embedding dimension.
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

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
                    f"Gemini embedding count mismatch: requested {len(texts)} texts "
                    f"but got {len(vectors)} vectors."
                )

            return np.array(vectors)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=gemini_embed_func,
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto",
        )

        logger.info("\nQuerying processed document (Gemini native SDK):")
        query = "このドキュメントの主な内容を要約してください。"
        result = await rag.aquery(query, mode="hybrid")
        logger.info(f"Answer: {result}")

    except Exception as e:
        logger.error(f"Error processing with RAG (Gemini): {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the Gemini example."""
    parser = argparse.ArgumentParser(description="RAGAnything Gemini Native SDK Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir",
        "-w",
        default="./rag_storage",
        help="Working directory path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory path",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key (defaults to GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview"),
        help="LLM model name",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview"),
        help="Embedding model name (Gemini)",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help=(
            "Parser selection. Built-ins: mineru, docling, paddleocr. "
            "Custom parsers registered by register_parser() are also accepted."
        ),
    )

    args = parser.parse_args()

    if not args.api_key:
        logger.error("Error: Gemini API key is required")
        logger.error("Set GEMINI_API_KEY environment variable or use --api-key option")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    asyncio.run(
        process_with_rag(
            file_path=args.file_path,
            output_dir=args.output,
            api_key=args.api_key,
            working_dir=args.working_dir,
            parser=args.parser,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
        )
    )


if __name__ == "__main__":
    configure_logging()

    print("RAGAnything Gemini Example")
    print("=" * 40)
    print("Processing document with Gemini native SDK multimodal RAG pipeline")
    print("=" * 40)

    main()
