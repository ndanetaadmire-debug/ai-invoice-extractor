"""Post-extraction verification. The model's output is a claim; these checks decide whether to believe it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from invx.schema import KEY_FIELDS, Invoice

VAT_RATE = Decimal("0.15")
CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Issue:
    field: str
    code: str
    message: str
    blocking: bool = True


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_issues(inv: Invoice, document: str) -> list[Issue]:
    """Every evidence snippet must literally occur in the document, and must contain the value.
    This is what catches an LLM that 'reads' a number that isn't on the page."""
    doc = _norm(document)
    issues: list[Issue] = []
    for name in KEY_FIELDS:
        field = getattr(inv, name)
        if field is None:
            continue
        ev = _norm(field.evidence)
        if not ev or ev not in doc:
            issues.append(
                Issue(name, "evidence_not_found", f"evidence {field.evidence!r} does not occur in the document")
            )
            continue
        if not _value_in_evidence(field.value, field.evidence):
            issues.append(
                Issue(name, "value_not_in_evidence", f"value {field.value!r} not supported by {field.evidence!r}")
            )
    return issues


def _value_in_evidence(value: object, evidence: str) -> bool:
    if isinstance(value, Decimal):
        # Collapse thousands separators ("1 899.00", "1,899.00", "1\u00a0899,00") before comparing.
        compact = re.sub(r"(?<=\d)[\s\u00a0,](?=\d{3}\b)", "", evidence)
        compact = re.sub(r"(?<=\d),(?=\d{2}\b)", ".", compact)
        return any(Decimal(n) == value for n in re.findall(r"\d+(?:\.\d+)?", compact))
    if hasattr(value, "isoformat"):  # dates are written many ways; require at least the day number
        day = value.day  # type: ignore[attr-defined]
        return re.search(rf"\b0?{day}\b", evidence) is not None
    return _norm(str(value)) in _norm(evidence)


def arithmetic_issues(inv: Invoice) -> list[Issue]:
    issues: list[Issue] = []
    sub, vat, total = inv.subtotal.value, inv.vat.value, inv.total.value
    if abs(sub + vat - total) > CENT:
        issues.append(Issue("total", "total_mismatch", f"{sub} + {vat} != {total}"))
    if abs(sub * VAT_RATE - vat) > Decimal("0.05"):
        issues.append(Issue("vat", "vat_rate", f"VAT {vat} is not 15% of {sub}", blocking=False))
    if inv.line_items:
        lines = sum((li.amount for li in inv.line_items), Decimal("0"))
        if abs(lines - sub) > CENT:
            issues.append(Issue("line_items", "lines_mismatch", f"line items sum to {lines}, subtotal is {sub}"))
        for i, li in enumerate(inv.line_items):
            if abs(li.quantity * li.unit_price - li.amount) > CENT:
                issues.append(Issue(f"line_items[{i}]", "line_math", f"{li.quantity} x {li.unit_price} != {li.amount}"))
    if inv.due_date and inv.due_date.value < inv.invoice_date.value:
        issues.append(Issue("due_date", "due_before_issue", "due date is before invoice date", blocking=False))
    return issues


def confidence(inv: Invoice, issues: list[Issue]) -> dict[str, float]:
    """Per-field confidence from verification, not from the model's self-report (which is poorly calibrated)."""
    bad = {i.field for i in issues if i.blocking}
    warn = {i.field for i in issues if not i.blocking}
    out: dict[str, float] = {}
    for name in KEY_FIELDS:
        if getattr(inv, name) is None:
            continue
        out[name] = 0.2 if name in bad else 0.7 if name in warn else 0.98
    return out
