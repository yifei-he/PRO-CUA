#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_entry(file_name: str) -> dict:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a minimal LlamaFactory dataset_info.json for a local dataset file."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        args.dataset_name: build_entry(args.dataset_file),
    }

    output_path = output_dir / "dataset_info.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(output_path)


if __name__ == "__main__":
    main()
