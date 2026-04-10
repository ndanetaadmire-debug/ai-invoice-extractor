from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from invx.cli import build_extractor
from invx.extract import Extractor, to_json


class Doc(BaseModel):
    text: str = Field(min_length=20, max_length=40_000, description="OCR text of the invoice")


def create_app(extractor: Extractor | None = None) -> FastAPI:
    ex = extractor or build_extractor()
    app = FastAPI(title="AI invoice extractor")

    @app.post("/extract")
    def extract(doc: Doc) -> dict[str, object]:
        res = ex.extract(doc.text)
        if res.status == "failed":
            raise HTTPException(422, detail={"errors": res.errors})
        out: dict[str, object] = json.loads(to_json(res))
        return out

    return app
