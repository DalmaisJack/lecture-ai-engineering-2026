"""宿題（発展）のハイブリッド検索。同梱DBのページ埋め込みを質問の代わりに使い、APIは呼ばない。"""

import json
import struct

import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import hybrid, tools
from app.agent import HYBRID_INSTRUCTIONS, agent
from app.config import RAG_DB
from app.retrieval import load_pages, open_index


@pytest.fixture(scope="module")
def like():
    with open_index(RAG_DB) as db:
        vectors = {row["page_id"]: list(struct.unpack(f"{len(row['embedding']) // 4}f", row["embedding"]))
                   for row in db.execute("SELECT page_id, embedding FROM chunks")}
    return lambda page_id: (lambda text, task: vectors[page_id])


def test_two_lectures_and_terms_get_entries(like):
    out = hybrid.search_hybrid("第2回のLoRAと、第3回のメモリは何が違いますか", embedder=like("lec03-main-p013"))
    ids = [r["page_id"] for r in out["results"]]
    assert len(ids) == len(set(ids)) <= hybrid.BASE_K + hybrid.EXTRA_MAX
    assert out["scope"] == "第2回・第3回" and {"LoRA", "メモリ"} <= set(out["terms"])
    assert {pid[:5] for pid in ids} == {"lec02", "lec03"}


def test_scope_word_boundary_and_same_section(like):
    out = hybrid.search_hybrid("第2回のQLoRAとは", embedder=like("lec02-main-p052"))
    assert all(r["page_id"].startswith("lec02-main") for r in out["results"])
    assert "QLoRA" in out["terms"] and "LoRA" not in out["terms"]
    sections = {p["page_id"]: p["section"] for p in load_pages()}
    top = out["results"][0]["page_id"]
    for r in out["results"]:
        if "のページ" in r["why"]:
            assert sections[r["page_id"]] == sections[top]


def test_hybrid_tool_reaches_the_model(monkeypatch, like):
    monkeypatch.setattr(hybrid, "embed_text", like("lec02-main-p052"))
    model = TestModel(call_tools=["search_hybrid"])
    with agent.override(model=model, instructions=HYBRID_INSTRUCTIONS, tools=[tools.search_hybrid]):
        result = agent.run_sync("LoRAとは？")
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns and json.loads(returns[0].content)["results"]
