from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Literal

from pydantic import ValidationError

from invx.checks import Issue, arithmetic_issues, confidence, evidence_issues
from invx.llm import LLM, LLMError, parse_json
from invx.schema import Invoice

SYSTEM = """You extract data from supplier invoices for an accounts-payable system.
Return JSON matching the schema. For every evidenced field, `evidence` must be copied verbatim from the document
(the shortest snippet that contains the value). Never infer or calculate a value that is not printed.
If an optional field is not on the invoice, use null. Amounts are numbers without currency symbols."""

# USD per 1M tokens (input, output). Kept in config, not code, in production.
PRICES: dict[str, tuple[float, float]] = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "rules": (0.0, 0.0)}

Status = Literal["auto_approved", "needs_review", "failed"]


@dataclass(slots=True)
class Result:
    status: Status
    invoice: Invoice | None
    issues: list[Issue] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    attempts: int = 0
    model: str | None = None
    cost_usd: float = 0.0
    cached: bool = False
    errors: list[str] = field(default_factory=list)


class Extractor:
    """LLM extraction with verification:
    1. structured output against the Pydantic schema
    2. repair loop: schema errors and failed checks are fed back verbatim (bounded)
    3. evidence verification: every key field's snippet must occur in the document and contain the value
    4. arithmetic and VAT checks
    5. model fallback chain on provider errors or exhausted repairs
    6. content-hash cache: the same document is never paid for twice
    Status is `auto_approved` only when nothing blocking remains; otherwise a human reviews it."""

    def __init__(self, models: list[LLM], *, max_repairs: int = 2, max_chars: int = 40_000) -> None:
        if not models:
            raise ValueError("at least one model is required")
        self.models = models
        self.max_repairs = max_repairs
        self.max_chars = max_chars
        self._cache: dict[str, Result] = {}
        self._schema = Invoice.model_json_schema()

    def extract(self, document: str) -> Result:
        key = hashlib.sha256(document.encode()).hexdigest()
        if key in self._cache:
            return replace(self._cache[key], cached=True, cost_usd=0.0)
        if len(document) > self.max_chars:
            return Result("failed", None, errors=[f"document exceeds {self.max_chars} characters"])

        total_cost, attempts, errors = 0.0, 0, []
        best: Result | None = None
        for model in self.models:
            feedback: str | None = None
            for _ in range(self.max_repairs + 1):
                attempts += 1
                user = f"Document:\n<<<\n{document}\n>>>" + (
                    f"\n\nYour previous answer had problems:\n{feedback}\nReturn corrected JSON." if feedback else ""
                )
                try:
                    out = model.complete(SYSTEM, user, self._schema)
                except LLMError as exc:
                    errors.append(f"{model.name}: {exc}")
                    break  # provider failure → next model
                total_cost += _cost(model.name, out.input_tokens, out.output_tokens)
                try:
                    inv = Invoice.model_validate(parse_json(out.text))
                except (ValueError, ValidationError) as exc:
                    feedback = _format_validation(exc)
                    errors.append(f"{model.name}: invalid output")
                    continue
                issues = evidence_issues(inv, document) + arithmetic_issues(inv)
                blocking = [i for i in issues if i.blocking]
                res = Result(
                    "needs_review" if blocking else "auto_approved",
                    inv,
                    issues,
                    confidence(inv, issues),
                    attempts,
                    model.name,
                    round(total_cost, 6),
                )
                if not blocking:
                    res.errors = errors  # keep the trail of earlier model failures
                    self._cache[key] = res
                    return res
                best = res if best is None or len(blocking) < len([i for i in best.issues if i.blocking]) else best
                feedback = "\n".join(f"- {i.field}: {i.message}" for i in blocking)
        if best:
            best.attempts, best.cost_usd, best.errors = attempts, round(total_cost, 6), errors
            self._cache[key] = best
            return best
        return Result("failed", None, attempts=attempts, cost_usd=round(total_cost, 6), errors=errors)


def _cost(model: str, inp: int, out: int) -> float:
    pi, po = PRICES.get(model, (0.0, 0.0))
    return (inp * pi + out * po) / 1_000_000


def _format_validation(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "\n".join(f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:10])
    return f"- {exc}"


def to_json(res: Result) -> str:
    def enc(o: object) -> object:
        if isinstance(o, Decimal):
            return str(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(type(o))

    payload = {
        "status": res.status,
        "model": res.model,
        "attempts": res.attempts,
        "cost_usd": res.cost_usd,
        "cached": res.cached,
        "confidence": res.confidence,
        "issues": [asdict(i) for i in res.issues],
        "invoice": res.invoice.model_dump() if res.invoice else None,
        "errors": res.errors,
    }
    return json.dumps(payload, default=enc, indent=2)
