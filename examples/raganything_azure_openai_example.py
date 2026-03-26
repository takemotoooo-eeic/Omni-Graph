#!/usr/bin/env python
"""
Azure OpenAI example script demonstrating parser integration with RAGAnything.

このサンプルでは Azure OpenAI の
- Embedding (AZURE_EMBEDDING_DEPLOYMENT)
- LLM (AZURE_CHEAP_LLM_DEPLOYMENT / AZURE_BEST_LLM_DEPLOYMENT)
を使って RAGAnything を実行する方法を示します。
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.azure_openai import azure_openai_complete_if_cache
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_azure_openai_example.log"))

    print(f"\nRAGAnything Azure OpenAI example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str | None = None,
    api_version: str | None = None,
    working_dir: str | None = None,
    parser: str | None = None,
):
    """
    Process document with RAGAnything using Azure OpenAI.

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: Azure OpenAI API key
        base_url: Azure OpenAI endpoint (例: https://xxxx.openai.azure.com/)
        api_version: Azure OpenAI API version (例: 2024-08-01-preview)
        working_dir: Working directory for RAG storage
        parser: Parser type (mineru, docling, paddleocr など)
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # Parser selection: mineru, docling, or paddleocr
            parse_method="auto",  # Parse method: auto, ocr, or txt
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Azure 用のデプロイメント名を環境変数から取得
        cheap_llm_deployment = os.getenv("AZURE_CHEAP_LLM_DEPLOYMENT", "gpt-4o-mini")
        best_llm_deployment = os.getenv("AZURE_BEST_LLM_DEPLOYMENT", "gpt-4o")

        # Define LLM model function (cheap model)
        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            return azure_openai_complete_if_cache(
                model=cheap_llm_deployment,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                api_version=api_version,
                **kwargs,
            )

        # Define vision model function for image processing (best model)
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if history_messages is None:
                history_messages = []

            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return azure_openai_complete_if_cache(
                    model=best_llm_deployment,
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    api_version=api_version,
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return azure_openai_complete_if_cache(
                    model=best_llm_deployment,
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    api_version=api_version,
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Embedding 関数は openai_embed の「中身」を直接呼び出す形で Azure 用にラップする
        # （openai_embed 自体も EmbeddingFunc なので、.func で素の実装を叩く）
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_deployment = os.getenv(
            "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        )

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts=texts,
                model=embedding_deployment,
                base_url=base_url,
                api_key=api_key,
                use_azure=True,
                azure_deployment=embedding_deployment,
                api_version=api_version,
            ),
        )

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document (Azure OpenAI):")

        # 1. Pure text queries using aquery()
        text_queries = [
            "このドキュメントの主な内容を要約してください。",
            "このドキュメントで議論されている重要なトピックを列挙してください。",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        logger.info(
            "\n[Multimodal Query]: ドキュメント内容と以下の性能比較表を踏まえて考察してください"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "この性能比較表の内容を、ドキュメント中の関連する記述と照らし合わせて分析してください。",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Processing_Time
                                RAGAnything,95.2%,120ms
                                Traditional_RAG,87.3%,180ms
                                Baseline,82.1%,200ms""",
                    "table_caption": "性能比較結果",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. Another multimodal query with equation content
        logger.info("\n[Multimodal Query]: 数式の説明と文書との関連付け")
        equation_result = await rag.aquery_with_multimodal(
            "次の数式を説明し、ドキュメント内の関連する指標や評価方法と関連付けて説明してください。",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1-score の計算式",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG (Azure OpenAI): {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the Azure OpenAI example"""
    parser = argparse.ArgumentParser(description="RAGAnything Azure OpenAI Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AZURE_OPENAI_API_KEY"),
        help="Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("AZURE_OPENAI_ENDPOINT"),
        help="Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)",
    )
    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        help="Azure OpenAI API version (defaults to AZURE_OPENAI_API_VERSION env var)",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help=(
            "Parser selection. Built-ins: mineru, docling, paddleocr. "
            "Custom parsers that you register via register_parser() in the "
            "same Python process are also accepted when using RAGAnything as "
            "a library. This example script does not perform any automatic "
            "plugin discovery."
        ),
    )

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error("Error: Azure OpenAI API key is required")
        logger.error(
            "Set AZURE_OPENAI_API_KEY environment variable or use --api-key option"
        )
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.api_version,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Azure OpenAI Example")
    print("=" * 40)
    print("Processing document with Azure OpenAI multimodal RAG pipeline")
    print("=" * 40)

    main()