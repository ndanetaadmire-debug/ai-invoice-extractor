"""A deterministic stand-in for an LLM that speaks the same JSON contract (values + evidence).
Used for offline demos, the CI accuracy gate, and as a zero-cost last-resort fallback model."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from invx.llm import Completion

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y")
_AMOUNT = r"(?:R|ZAR)?\s?([\d][\d\s,]*[.,]\d{2})"


def _find(pattern: str, text: str) -> tuple[str, str] | None:
    m = re.search(pattern, text, re.I | re.M)
    return (m.group(1).strip(), m.group(0).strip()) if m else None


def _amount(raw: str) -> str:
    raw = re.sub(r"[\s]", "", raw)
    if re.search(r",\d{2}$", raw):
        raw = raw.replace(".", "").replace(",", ".")
    return raw.replace(",", "")


def _date(raw: str) -> str | None:
    for f in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), f).date().isoformat()
        except ValueError:
            continue
    return None


class RulesModel:
    name = "rules"

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> Completion:
        doc = user.split("<<<", 1)[-1].split(">>>", 1)[0]
        out: dict[str, Any] = {"currency": "ZAR", "line_items": []}

        def ev(key: str, pattern: str, conv: Any = str) -> None:
            hit = _find(pattern, doc)
            if hit and (value := conv(hit[0])) is not None:
                out[key] = {"value": value, "evidence": hit[1]}

        ev("supplier_name", r"^(?:from|supplier|vendor|bill from)[:\s]+(.+)$")
        ev("supplier_vat_number", r"vat\s*(?:reg(?:istration)?\.?\s*)?(?:no\.?|number|#)[:\s]*(4\d{9})")
        ev("invoice_number", r"(?:tax\s+)?invoice\s*(?:no\.?|number|#)[:\s]*([A-Z0-9][A-Z0-9/-]+)")
        ev(
            "invoice_date",
            r"(?:invoice\s+)?date[:\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2} \w+ \d{4})",
            _date,
        )
        ev("due_date", r"due(?:\s+date)?[:\s]+(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2} \w+ \d{4})", _date)
        ev("subtotal", r"sub\s*-?\s*total(?:\s+excl\.?\s+vat)?[:\s]+" + _AMOUNT, _amount)
        ev("vat", r"^\s*vat(?:\s*@?\s*\(?15\s?%\)?)?[:\s]+" + _AMOUNT, _amount)
        ev("total", r"^\s*(?:total(?:\s+due|\s+incl\.?\s+vat)?|amount\s+due|balance\s+due)[:\s]+" + _AMOUNT, _amount)
        if m := re.search(r"currency[:\s]+([A-Z]{3})", doc):
            out["currency"] = m.group(1)

        for m in re.finditer(
            r"^\s*(.+?)\s{2,}(\d+(?:\.\d+)?)\s*x\s*" + _AMOUNT + r"\s{2,}" + _AMOUNT + r"\s*$", doc, re.M
        ):
            out["line_items"].append(
                {
                    "description": m.group(1).strip(),
                    "quantity": m.group(2),
                    "unit_price": _amount(m.group(3)),
                    "amount": _amount(m.group(4)),
                }
            )
        text = json.dumps(out)
        return Completion(text, input_tokens=len(user) // 4, output_tokens=len(text) // 4, model=self.name)
