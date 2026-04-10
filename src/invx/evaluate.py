"""Field-level accuracy against labelled invoices. Run in CI so prompt/model changes can't silently regress."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from invx.extract import Extractor
from invx.schema import KEY_FIELDS


@dataclass(frozen=True, slots=True)
class Accuracy:
    field_accuracy: float
    auto_approval_rate: float
    per_field: dict[str, float]
    errors: list[str]


def _norm(v: object) -> object:
    if isinstance(v, Decimal):
        return str(v.quantize(Decimal("0.01")))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def evaluate(extractor: Extractor, data_dir: Path) -> Accuracy:
    truth: dict[str, dict[str, object]] = json.loads((data_dir / "truth.json").read_text())
    hits: dict[str, list[bool]] = {f: [] for f in (*KEY_FIELDS, "lines")}
    errors: list[str] = []
    approved = 0
    for fname, expected in truth.items():
        res = extractor.extract((data_dir / "invoices" / fname).read_text())
        approved += res.status == "auto_approved"
        inv = res.invoice
        for f in KEY_FIELDS:
            got = None if inv is None or getattr(inv, f) is None else _norm(getattr(inv, f).value)
            ok = got == expected.get(f)
            hits[f].append(ok)
            if not ok:
                errors.append(f"{fname}.{f}: expected {expected.get(f)!r}, got {got!r}")
        ok_lines = inv is not None and len(inv.line_items) == expected["lines"]
        hits["lines"].append(ok_lines)
        if not ok_lines:
            errors.append(f"{fname}.lines: expected {expected['lines']}, got {len(inv.line_items) if inv else None}")
    per = {f: round(sum(v) / len(v), 3) for f, v in hits.items()}
    allv = [x for v in hits.values() for x in v]
    return Accuracy(round(sum(allv) / len(allv), 3), round(approved / len(truth), 3), per, errors)
