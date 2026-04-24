"""Entry point for running the neural memory MCP server."""

import argparse
import asyncio
import os


def run():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neural-memory",
        description="Neural Memory MCP Server — human-like memory system for LLMs",
    )
    parser.add_argument(
        "--storage-dir",
        help="Data storage directory (overrides config.yaml and NEURAL_MEMORY_STORAGE_DIR)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml (overrides NEURAL_MEMORY_CONFIG)",
    )
    parser.add_argument(
        "--embedding-model",
        help="Sentence-transformers model name",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    # CLI args → environment variables (highest priority)
    if args.storage_dir:
        os.environ["NEURAL_MEMORY_STORAGE_DIR"] = args.storage_dir
    if args.config:
        os.environ["NEURAL_MEMORY_CONFIG"] = args.config
    if args.embedding_model:
        os.environ["NEURAL_MEMORY_EMBEDDING_MODEL"] = args.embedding_model
    if args.log_level:
        os.environ["NEURAL_MEMORY_LOG_LEVEL"] = args.log_level

    from neural_memory.server import main

    asyncio.run(main())


if __name__ == "__main__":
    run()
