"""invx extract FILE [--model rules|openai] | invx eval [--min-accuracy 0.95]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from invx.evaluate import evaluate
from invx.extract import Extractor, to_json
from invx.llm import LLM, OpenAICompatibleLLM
from invx.rules_model import RulesModel


def build_extractor() -> Extractor:
    models: list[LLM] = []
    if key := os.getenv("LLM_API_KEY"):
        base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        models.append(OpenAICompatibleLLM(base, key, os.getenv("LLM_MODEL", "gpt-4.1-mini")))
        if fb := os.getenv("LLM_FALLBACK_MODEL"):
            models.append(OpenAICompatibleLLM(base, key, fb))
    models.append(RulesModel())  # zero-cost last resort; also the offline default
    return Extractor(models)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="invx")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("file", type=Path)
    v = sub.add_parser("eval")
    v.add_argument("--data", type=Path, default=Path("data"))
    v.add_argument("--min-accuracy", type=float, default=0.95)
    a = p.parse_args(argv)
    ex = build_extractor()
    if a.cmd == "extract":
        res = ex.extract(a.file.read_text())
        print(to_json(res))
        return 0 if res.status != "failed" else 1
    r = evaluate(ex, a.data)
    print(f"field accuracy {r.field_accuracy:.1%} · auto-approved {r.auto_approval_rate:.0%}")
    for f, acc in r.per_field.items():
        print(f"  {f:22} {acc:.0%}")
    for err in r.errors:
        print(f"  ✗ {err}")
    return 0 if r.field_accuracy >= a.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
