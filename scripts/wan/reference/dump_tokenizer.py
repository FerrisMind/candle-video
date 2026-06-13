#!/usr/bin/env python3
"""Dump Wan tokenizer ids for Rust parity tests."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

try:
    import ftfy

    def basic_clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

except ImportError:

    def basic_clean(text: str) -> str:
        text = html.unescape(html.unescape(text))
        return text.strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=226)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    clean = prompt_clean(args.prompt)
    out = tok(
        clean,
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    fixture = {
        "prompt": args.prompt,
        "clean_prompt": clean,
        "max_length": args.max_length,
        "input_ids": out["input_ids"],
        "attention_mask": out["attention_mask"],
        "seq_len": sum(out["attention_mask"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2))
    print(f"wrote {args.output} seq_len={fixture['seq_len']}")


if __name__ == "__main__":
    main()
