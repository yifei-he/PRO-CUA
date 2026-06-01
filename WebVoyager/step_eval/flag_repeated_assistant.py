"""
Scan interact_messages_with_prm_score.json files for assistant turns whose
content appears >= min_repeat times anywhere in the conversation (non-consecutive)
and set their score to 0 with an auto-judge note.

Usage:
    python flag_repeated_assistant.py \
        --root /scratch/yifeihe/cua/data/qwen3_WebVoyager_train_data_unique_ids_train/round_1 \
        --min-repeat 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

AUTO_JUDGE_MSG = (
    "Auto score: repeated identical assistant content (>= {min_repeat} occurrences)."
)


def find_repeated_idxs(conversations: List[Dict], min_repeat: int) -> Set[int]:
    """Return indices of assistant turns whose content appears at least min_repeat times."""
    content_to_idxs: Dict[str, List[int]] = {}
    for i, turn in enumerate(conversations):
        if turn.get("from") == "assistant":
            content = turn.get("value", "")
            content_to_idxs.setdefault(content, []).append(i)

    repeated_idxs: Set[int] = set()
    for idxs in content_to_idxs.values():
        if len(idxs) >= min_repeat:
            repeated_idxs.update(idxs)
    return repeated_idxs


def process_file(path: Path, min_repeat: int, dry_run: bool = False) -> int:
    """Update a single JSON file if repeated assistant content is found.

    Returns number of turns modified.
    """
    with path.open() as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    repeated_idxs = find_repeated_idxs(conversations, min_repeat=min_repeat)
    if not repeated_idxs:
        return 0

    for idx in repeated_idxs:
        turn = conversations[idx]
        turn["score"] = 0
        turn["judge"] = AUTO_JUDGE_MSG.format(min_repeat=min_repeat)

    if not dry_run:
        with path.open("w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    return len(repeated_idxs)


def iter_target_files(root: Path):
    yield from root.rglob("interact_messages_with_prm_score.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-repeat", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes.")
    args = parser.parse_args()

    total_files = 0
    touched_files = 0
    total_turns = 0

    for path in iter_target_files(args.root):
        total_files += 1
        modified = process_file(path, min_repeat=args.min_repeat, dry_run=args.dry_run)
        if modified:
            touched_files += 1
            total_turns += modified
            status = "DRY" if args.dry_run else "UPDATED"
            print(f"[{status}] {path} -> {modified} turns")

    print(
        f"Scanned {total_files} files. "
        f"Modified {touched_files} files, {total_turns} assistant turns."
    )


if __name__ == "__main__":
    main()
