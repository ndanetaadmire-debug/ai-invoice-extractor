"""The extraction contract. Money is Decimal; every key field carries the exact text it came from."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Evidenced[T](BaseModel):
    """A value plus the verbatim snippet of the document it was read from."""

    model_config = ConfigDict(frozen=True)
    value: T
    evidence: str = Field(description="Exact substring of the document this value was read from")


class LineItem(BaseModel):
    description: str
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    amount: Decimal = Field(ge=0)


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier_name: Evidenced[str]
    supplier_vat_number: Evidenced[str] | None = None
    invoice_number: Evidenced[str]
    invoice_date: Evidenced[date]
    due_date: Evidenced[date] | None = None
    currency: str = Field(default="ZAR", pattern=r"^[A-Z]{3}$")
    subtotal: Evidenced[Decimal]
    vat: Evidenced[Decimal]
    total: Evidenced[Decimal]
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("supplier_vat_number")
    @classmethod
    def sa_vat_format(cls, v: Evidenced[str] | None) -> Evidenced[str] | None:
        if v is not None and not (len(v.value) == 10 and v.value.isdigit() and v.value.startswith("4")):
            raise ValueError("South African VAT numbers are 10 digits starting with 4")
        return v


KEY_FIELDS = (
    "supplier_name",
    "supplier_vat_number",
    "invoice_number",
    "invoice_date",
    "due_date",
    "subtotal",
    "vat",
    "total",
)
