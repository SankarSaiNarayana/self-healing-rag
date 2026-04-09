#!/usr/bin/env python3
"""Run a small golden set against /query (server must be up)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--collection", default="docs")
    p.add_argument("--user_id", default="eval")
    args = p.parse_args()

    questions = [
        "What is this system?",
        "What did I ask before?",
    ]
    base = args.base.rstrip("/")
    for q in questions:
        payload = json.dumps(
            {"question": q, "collection": args.collection, "user_id": args.user_id, "return_context": False}
        ).encode()
        req = urllib.request.Request(
            f"{base}/query",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print("FAIL", q, e, file=sys.stderr)
            continue
        print("---")
        print("Q:", q)
        print("confidence:", data.get("confidence"), "used_llm:", data.get("used_llm"))
        print("warnings:", data.get("warnings"))


if __name__ == "__main__":
    main()
