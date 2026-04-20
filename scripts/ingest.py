from __future__ import annotations

import argparse
import os

from core.ingest_service import ingest_from_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the configured vector store")
    parser.add_argument("--data_dir", required=True, help="Directory containing files to ingest")
    parser.add_argument("--collection", default="docs", help="Collection / index name")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "900")))
    parser.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "150")))
    args = parser.parse_args()

    n = ingest_from_directory(
        data_dir=args.data_dir,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingested {n} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
