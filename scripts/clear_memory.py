"""Clear episodic + semantic memory and procedural stats for a user (does not delete document index)."""

from __future__ import annotations

import argparse

from core.memory.clear_memory import clear_user_memory


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user_id", default="demo", help="User id whose memory to clear")
    args = p.parse_args()
    out = clear_user_memory(user_id=args.user_id)
    print(out)


if __name__ == "__main__":
    main()
