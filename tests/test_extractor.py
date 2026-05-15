import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invx.api import create_app
from invx.checks import arithmetic_issues, evidence_issues
from invx.cli import main
from invx.evaluate import evaluate
from invx.extract import Extractor
from invx.llm import Completion, LLMError
from invx.rules_model import RulesModel
from invx.schema import Invoice

DOC = Path("data/invoices/acme.txt").read_text()


def good_json(**over: Any) -> dict[str, Any]:
    base = json.loads(RulesModel().complete("", f"<<<{DOC}>>>", {}).text)
    base.update(over)
    return base


class Scripted:
    """Fake LLM that replays outputs (dicts, raw strings or exceptions) and records prompts."""

    def __init__(self, name: str, outputs: list[Any]) -> None:
        self.name, self.outputs, self.prompts = name, outputs, []

    def complete(self, system: str, user: str, json_schema: dict[str, Any]) -> Completion:
        self.prompts.append(user)
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        return Completion(o if isinstance(o, str) else json.dumps(o), 1000, 200, self.name)


def test_happy_path_auto_approves_with_cost() -> None:
    m = Scripted("gpt-4.1-mini", [good_json()])
    r = Extractor([m]).extract(DOC)
    assert r.status == "auto_approved" and r.attempts == 1 and r.model == "gpt-4.1-mini"
    assert r.invoice and str(r.invoice.total.value) == "8050.00"
    assert r.cost_usd == pytest.approx((1000 * 0.40 + 200 * 1.60) / 1e6)
    assert min(r.confidence.values()) > 0.9


def test_repair_loop_feeds_back_schema_errors() -> None:
    bad = good_json(supplier_vat_number={"value": "12345", "evidence": "VAT Reg No: 4123456789"})
    m = Scripted("gpt-4.1-mini", ["not json at all", bad, good_json()])
    r = Extractor([m]).extract(DOC)
    assert r.status == "auto_approved" and r.attempts == 3
    assert "no JSON object" in m.prompts[1] and "10 digits starting with 4" in m.prompts[2]


def test_hallucinated_value_is_caught_by_evidence_check_and_repaired() -> None:
    invented = good_json(total={"value": "9050.00", "evidence": "Total due: R9,050.00"})
    m = Scripted("gpt-4.1-mini", [invented, good_json()])
    r = Extractor([m]).extract(DOC)
    assert r.status == "auto_approved" and r.attempts == 2
    assert "does not occur in the document" in m.prompts[1]


def test_persistent_problem_goes_to_human_review_not_auto_approval() -> None:
    wrong = good_json(vat={"value": "1150.00", "evidence": "VAT (15%): R1,050.00"})
    m = Scripted("gpt-4.1-mini", [wrong, wrong, wrong])
    r = Extractor([m], max_repairs=2).extract(DOC)
    assert r.status == "needs_review"
    assert {i.code for i in r.issues} >= {"value_not_in_evidence", "total_mismatch"}
    assert r.confidence["vat"] < 0.5


def test_provider_failure_falls_back_to_next_model() -> None:
    primary = Scripted("gpt-4.1", [LLMError("503 Service Unavailable")])
    r = Extractor([primary, RulesModel()]).extract(DOC)
    assert r.status == "auto_approved" and r.model == "rules"
    assert any("503" in e for e in r.errors)


def test_cache_means_same_document_costs_once() -> None:
    m = Scripted("gpt-4.1-mini", [good_json()])
    ex = Extractor([m])
    first, second = ex.extract(DOC), ex.extract(DOC)
    assert second.cached and second.cost_usd == 0 and first.invoice == second.invoice
    assert len(m.prompts) == 1


def test_limits_and_total_failure() -> None:
    assert Extractor([RulesModel()], max_chars=10).extract(DOC).status == "failed"
    dead = Scripted("x", [LLMError("down")])
    assert Extractor([dead]).extract(DOC).status == "failed"
    with pytest.raises(ValueError):
        Extractor([])


def test_checks_directly() -> None:
    inv = Invoice.model_validate(
        good_json(line_items=[{"description": "x", "quantity": "2", "unit_price": "10", "amount": "25"}])
    )
    codes = {i.code for i in arithmetic_issues(inv)}
    assert {"lines_mismatch", "line_math"} <= codes
    assert evidence_issues(inv, DOC) == []
    assert evidence_issues(inv, "a totally different document")[0].code == "evidence_not_found"


def test_accuracy_gate_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    r = evaluate(Extractor([RulesModel()]), Path("data"))
    assert r.field_accuracy == 1.0 and r.auto_approval_rate == 1.0, r.errors
    assert main(["eval"]) == 0
    assert main(["extract", "data/invoices/karoo.txt"]) == 0
    assert '"status": "auto_approved"' in capsys.readouterr().out


def test_api() -> None:
    c = TestClient(create_app(Extractor([RulesModel()])))
    body = c.post("/extract", json={"text": Path("data/invoices/hosting.txt").read_text()}).json()
    assert body["status"] == "auto_approved" and body["invoice"]["total"]["value"] == "2298.85"
    assert c.post("/extract", json={"text": "short"}).status_code == 422
