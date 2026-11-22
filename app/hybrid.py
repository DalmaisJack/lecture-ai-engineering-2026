"""第5回 宿題（発展）：通常の検索に、目次（回・章）と用語のグラフで入口を足すハイブリッド検索。

1. 範囲：質問に「第N回」があれば、その回のページだけを候補にする（目次の上の階層で絞る）
2. 土台：範囲の中で質問に近い上位5件。通常の検索（RAG_TOP_K=5）と同じ件数で、取りこぼしを増やさない
3. 足す：次の順に、合計3件まで（話題が1つに偏るのを防ぐ）
   (1) 質問に出てくる回ごとの入口：その回で質問に最も近いページ（2つ以上の回が出てくるとき）
   (2) 質問に出てくる用語ごとの入口：その用語に言及する（mentions）ページのうち、
       見出しにその用語があるページを優先し、質問に最も近いもの。言及ページが少ない用語から2件まで
   (3) 1位のページと同じ章の前後ページ（next_page）。範囲の中で質問に近い順15位以内のものだけ
   区切りのスライド（本文150字未満）は、足す候補にしない
質問の埋め込みは1回だけ。ページの埋め込みは同梱の検索DBのものを使うので、索引の再作成は不要。
"""

import json
import re
import unicodedata
from functools import cache

import sqlite_vec

from app.config import RAG_DB, RAG_GRAPH, RAG_TOP_K
from app.retrieval import embed_text, load_pages, open_index

BASE_K = RAG_TOP_K    # 土台にする、質問に近い順の件数
EXTRA_MAX = 3         # 回・用語・前後ページで足すページの上限
TERM_MAX = 2          # そのうち、用語の入口として足す上限
NEIGHBOR_RANK = 15    # 前後ページを足す条件：範囲の中で、質問に近い順にこの順位以内
MIN_CHARS = 150       # これより短い本文（区切りのスライド）は、足す候補にしない


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _is_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def _mentions(text: str, name: str) -> bool:
    """英数字の名前は単語の境目で照合する（LoRA が QLoRA の中で当たらないように）。"""
    name = _norm(name).strip()
    if not name:
        return False
    if not name.isascii():
        return name in text
    for m in re.finditer(r"\s*".join(map(re.escape, name.split())), text):
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        if not _is_word_char(before) and not _is_word_char(after):
            return True
    return False


@cache
def _graph() -> tuple[tuple[dict, ...], dict, dict]:
    """用語ノード、用語→言及するページ（mentions）、前後ページ（next_page）を読み込む。"""
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    terms = tuple(node for node in graph["nodes"] if node["kind"] == "term")
    term_pages, neighbors = {}, {}
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] == "mentions":
            term_pages.setdefault(b, set()).add(a)
        elif edge["relation"] == "next_page":
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    return terms, term_pages, neighbors


def _names(term: dict) -> list[str]:
    return [term["title"], *term.get("aliases", [])]


def lectures_in(query: str, available: set[int]) -> list[int]:
    """質問に書かれた「第N回」のうち、検索対象にある回を返す。"""
    found = {int(n) for n in re.findall(r"第\s*(\d+)\s*回", _norm(query))}
    return sorted(found & available)


def terms_in(query: str) -> list[dict]:
    """質問に名前か別名が出てくる用語ノードを返す。"""
    text = _norm(query)
    return [term for term in _graph()[0] if any(_mentions(text, name) for name in _names(term))]


def search_hybrid(query: str, embedder=None) -> dict:
    """質問を1回だけ埋め込んで全ページとの距離を出し、土台の5件に入口と前後ページを足す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    query = query.strip()
    pages = {page["page_id"]: page for page in load_pages()}
    with open_index(RAG_DB) as db:
        vector = (embedder or embed_text)(query, "RETRIEVAL_QUERY")
        rows = db.execute("SELECT page_id, vec_distance_cosine(embedding, ?) AS distance FROM chunks",
                          (sqlite_vec.serialize_float32(vector),)).fetchall()
    distance = {row["page_id"]: row["distance"] for row in rows if row["page_id"] in pages}

    # 1. 範囲
    lectures = lectures_in(query, {page["lecture"] for page in pages.values()})
    ranked = sorted((pid for pid in distance if not lectures or pages[pid]["lecture"] in lectures),
                    key=lambda pid: (distance[pid], pid))
    rank = {pid: i for i, pid in enumerate(ranked, 1)}
    usable = {pid for pid in ranked if len(pages[pid]["content"]) >= MIN_CHARS}
    chosen: dict[str, list[str]] = {}
    added = {"total": 0, "term": 0}

    def cite(pid: str) -> str:
        return f"第{pages[pid]['lecture']}回・PDF p.{pages[pid]['page']}"

    def add(pid: str, reason: str, kind: str) -> None:
        if pid in chosen:                  # すでに選んだページには、選んだ理由だけを足す
            chosen[pid].append(reason)
        elif added["total"] < EXTRA_MAX and (kind != "term" or added["term"] < TERM_MAX):
            chosen[pid] = [reason]
            added["total"] += 1
            added["term"] += kind == "term"

    # 2. 土台
    for pid in ranked[:BASE_K]:
        chosen[pid] = [f"質問に近い順{rank[pid]}位"]

    # 3-(1) 回ごとの入口
    if len(lectures) >= 2:
        for lecture in lectures:
            best = next((pid for pid in ranked if pid in usable and pages[pid]["lecture"] == lecture), None)
            if best:
                add(best, f"第{lecture}回の入口", "lecture")

    # 3-(2) 用語ごとの入口
    _, term_pages, neighbors = _graph()
    matched = []
    for term in terms_in(query):
        candidates = [pid for pid in term_pages.get(term["id"], ()) if pid in usable]
        if not candidates:
            continue
        titled = [pid for pid in candidates
                  if any(_mentions(_norm(pages[pid]["title"]), name) for name in _names(term))]
        best = min(titled or candidates, key=lambda pid: (distance[pid], pid))
        matched.append((len(candidates), term["id"], term["title"], best))
    for _, _, title, best in sorted(matched):
        add(best, f"用語「{title}」の入口", "term")

    # 3-(3) 1位のページと同じ章の前後ページ
    if ranked:
        top = ranked[0]
        sides = sorted((pid for pid in neighbors.get(top, ())
                        if pid in usable and rank[pid] <= NEIGHBOR_RANK
                        and pages[pid]["section"] == pages[top]["section"]),
                       key=lambda pid: (distance[pid], pid))
        for pid in sides:
            add(pid, f"{cite(top)}の{'次' if pages[pid]['page'] > pages[top]['page'] else '前'}のページ", "neighbor")

    return {
        "scope": "・".join(f"第{n}回" for n in lectures) if lectures else "第1〜3回",
        "terms": [title for _, _, title, _ in sorted(matched)],
        "results": [{"cite": cite(pid), "page_id": pid, "section": pages[pid]["section"],
                     "title": pages[pid]["title"], "why": "、".join(reasons),
                     "content": pages[pid]["content"]} for pid, reasons in chosen.items()],
    }
