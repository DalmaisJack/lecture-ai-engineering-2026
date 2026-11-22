"""第5回 宿題（発展）：同じ質問・モデル・資料で方式を比べ、根拠・時間・トークン数・費用を記録する。

    uv run python -m scripts.compare_retrieval --quick      # 試作品：3問 × 2方式 × 1回（数分）
    uv run python -m scripts.compare_retrieval              # 本番：5問 × 3方式 × 3回（10〜30分）
    uv run python -m scripts.compare_retrieval --selfcheck  # APIを使わず、仕組みだけ確かめる

止まっても、同じコマンドをもう一度実行すれば続きから再開する。
結果は lectures/05/hw/ に保存する（runs.jsonl：全回答と記録、summary.md：集計）。
"""

import argparse
import json
import re
import sys
import tempfile
import time
import unicodedata
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from app import agent as A
from app import tools
from app.config import MODEL, api_error_message

OUT = Path(__file__).resolve().parents[1] / "lectures/05/hw"

# evidence：根拠ページの組。組の中のどれか1ページがあれば、その組を満たす（別の書き方の同じ根拠）
# keywords：要点ごとの語句。どれか1つが回答にあれば、その要点を満たす（正しさの目安）
QUESTIONS = [
    {"id": "Q1", "kind": "1ページ（演習の質問）",
     "question": "第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？",
     "evidence": [["lec02-main-p052"]],
     "keywords": [["固定"], ["差分", "小さな行列", "低ランク", "AとB", "A・B"], ["4bit", "4ビット", "圧縮", "量子化"]]},
    {"id": "Q2", "kind": "複数ページ（problem.mdの例）",
     "question": "第3回の例では、コンパクションとセッションをまたぐメモリはどう違いますか。日々のメモはすべて長期メモリに入りますか",
     "evidence": [["lec03-main-p013"], ["lec03-main-p039"], ["lec03-main-p040"]],
     "keywords": [["要約"], ["セッションをまたぐ", "次回", "外の保存先", "ファイル"],
                  ["基準を超え", "全部は残さない", "すべては入らない", "残すものだけ", "一部だけ", "入りません", "入らない"]]},
    {"id": "Q3", "kind": "回をまたぐ説明",
     "question": "第2回で見た「長い文脈を均等には使えない」という性質は、第3回のエージェントではどんな問題として現れ、どう対処しますか？",
     "evidence": [["lec02-main-p015"], ["lec03-main-p003", "lec02-main-p062"],
                  ["lec03-main-p038", "lec03-main-p039", "lec03-main-p040"]],
     "keywords": [["見落と", "途中の情報", "真ん中"], ["メモリ"], ["要約", "コンパクション", "整理"]]},
    {"id": "Q4", "kind": "回をまたぐ用語の比較",
     "question": "第2回のLoRAによるファインチューニングと、第3回のメモリは、どちらも「覚えさせる」方法に見えます。何をどこに残すのかの違いを説明してください。",
     "evidence": [["lec02-main-p052", "lec02-main-p051"], ["lec03-main-p013", "lec03-main-p039"]],
     "keywords": [["重み", "パラメータ", "差分"], ["保存先", "ファイル", "書き出"], ["コンテキスト", "読み戻", "戻す", "読み込"]]},
    {"id": "Q5", "kind": "資料にない",
     "question": "今回作ったRAGの実測平均応答時間と正答率を教えてください",
     "evidence": [], "keywords": [], "unanswerable": True},
]
QUICK = ["Q1", "Q3", "Q4"]
METHODS = {"rag": "基本RAG", "graph": "Graph（元）", "tree": "Tree（元）", "hybrid": "ハイブリッド（改良）"}
DECLINE = ["確認でき", "記載がな", "記載はな", "記載されてい", "見つから", "含まれてい", "載ってい", "ありません"]


def mode(method: str):
    """agent.py の main() と同じ切り替え。"""
    target = A.agent
    if method == "graph":
        return target.override(instructions=A.GRAPH_INSTRUCTIONS, tools=[tools.search_graph])
    if method == "tree":
        return target.override(instructions=A.TREE_INSTRUCTIONS, tools=[tools.browse_tree])
    if method == "hybrid":
        return target.override(instructions=A.HYBRID_INSTRUCTIONS, tools=[tools.search_hybrid])
    return nullcontext()


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def cited_pages(answer: str) -> list[str]:
    """回答中の「第N回・PDF p.N」（p.39、40 や p.39〜41 の形も）を、ページIDにする。"""
    text = unicodedata.normalize("NFKC", answer)
    found = []
    for m in re.finditer(r"第\s*(\d+)\s*回[^第\n。]{0,15}?p\.?\s*(\d+)((?:\s*(?:[、,・/~〜\-–]|と|および)\s*(?:p\.?\s*)?\d+)*)", text):
        lecture, numbers = int(m.group(1)), [int(m.group(2))]
        for sep, n in re.findall(r"([、,・/~〜\-–]|と|および)\s*(?:p\.?\s*)?(\d+)", m.group(3)):
            n = int(n)
            numbers += list(range(numbers[-1] + 1, n + 1)) if sep in "~〜-–" and 0 < n - numbers[-1] <= 10 else [n]
        found += [f"lec{lecture:02d}-main-p{n:03d}" for n in numbers]
    return list(dict.fromkeys(found))


def run_once(method: str, q: dict, model) -> dict:
    started = perf_counter()
    with mode(method):
        result = A.agent.run_sync(q["question"], model=model)
    seconds = perf_counter() - started
    calls, retrieved, errors = [], [], []
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls.append({"tool": part.tool_name, "args": part.args_as_dict()})
            elif isinstance(part, ToolReturnPart):
                try:
                    data = json.loads(part.content) if isinstance(part.content, str) else dict(part.content)
                except (TypeError, ValueError):
                    continue
                if data.get("error"):
                    errors.append(str(data["error"]))
                for item in data.get("results", []) + data.get("seeds", []) + data.get("expanded", []):
                    retrieved.append(item["page_id"])
                node = data.get("node") or {}
                if node.get("kind") == "page":
                    retrieved.append(node["id"])
    usage = result.usage
    answer = str(result.output)
    retrieved = list(dict.fromkeys(retrieved))
    cited = cited_pages(answer)
    record = {
        "model": MODEL, "qid": q["id"], "method": method, "question": q["question"], "answer": answer,
        "calls": calls, "tool_errors": errors, "retrieved": retrieved, "cited": cited,
        "cited_not_retrieved": [p for p in cited if p not in retrieved],
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "requests": usage.requests, "cost": None if usage.cost is None else float(usage.cost),
        "seconds": round(seconds, 2),
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    groups = q["evidence"]
    record["evidence_retrieved"] = sum(bool(set(g) & set(retrieved)) for g in groups)
    record["evidence_cited"] = sum(bool(set(g) & set(cited)) for g in groups)
    flat = _flat(answer)
    record["keywords"] = sum(any(_flat(k) in flat for k in g) for g in q["keywords"])
    record["declined"] = any(w in flat for w in DECLINE) if q.get("unanswerable") else None
    return record


class RetryLater(Exception):
    pass


def run_with_retry(method: str, q: dict, model) -> dict:
    for attempt in range(4):
        try:
            record = run_once(method, q, model)
            if any(("レート制限" in e or "混み合" in e) for e in record["tool_errors"]):
                raise RetryLater("検索（埋め込み）の呼び出しが制限された")
            fatal = [e for e in record["tool_errors"] if "APIキー" in e or "APIエラー" in e or "検索DB" in e]
            if fatal:
                raise SystemExit(f"検索ツールでエラーが出たため止めます：{fatal[0]}")
            return record
        except (ModelHTTPError, RetryLater) as e:
            status = getattr(e, "status_code", 429)
            if status not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            wait = 65 if status == 429 else 30
            print(f"  [待機] {e.__class__.__name__}（{status}）。{wait}秒待って再実行します", flush=True)
            time.sleep(wait)


# ---------- 集計 ----------

def label(page_id: str) -> str:
    m = re.match(r"lec(\d+)-main-p(\d+)", page_id)
    return f"{int(m.group(1))}-{int(m.group(2))}" if m else page_id


def spread(values, fmt="{:,.0f}") -> str:
    if not values:
        return "—"
    if len(values) == 1 or min(values) == max(values):
        return fmt.format(mean(values))
    return f"{fmt.format(mean(values))}（{fmt.format(min(values))}〜{fmt.format(max(values))}）"


def summarize(records: list[dict], questions: list[dict], methods: list[str], trials: int) -> str:
    by = {}
    for r in records:
        by.setdefault((r["qid"], r["method"]), []).append(r)
    lines = [
        "# 第5回 宿題（発展）比較結果",
        "",
        f"- 回答モデル：{MODEL}／対象資料：第1〜3回の本講義214ページ（全方式で同じ）／各方式 {trials} 回",
        "- 根拠の取得・引用：質問ごとに決めた根拠ページの組のうち、ツールが本文を返した組・回答に「第N回・PDF p.N」が出た組の数（各回）",
        "- 要点の語句：要点ごとに決めた語句が回答に含まれた数（各回）。正しさの目安で、最後は回答を読んで確かめる",
        "- 数値は平均。カッコ内は最小〜最大。費用は生成APIの推定で、埋め込みは含まない",
        "",
        "| 質問 | 方式 | 根拠の取得 | 根拠の引用 | 要点の語句 | 取得していないページの引用 | 検索回数 | 取得ページ数 | 応答回数 | 入力トークン | 出力トークン | 時間（秒） | 推定費用（USD） |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for q in questions:
        n_ev, n_kw = len(q["evidence"]), len(q["keywords"])
        for method in methods:
            rs = by.get((q["id"], method), [])
            if not rs:
                continue
            if q.get("unanswerable"):
                ev_r = ev_c = "—"
                kw = f"確認できないと回答 {sum(r['declined'] for r in rs)}/{len(rs)}回"
            else:
                ev_r = "・".join(f"{r['evidence_retrieved']}/{n_ev}" for r in rs)
                ev_c = "・".join(f"{r['evidence_cited']}/{n_ev}" for r in rs)
                kw = "・".join(f"{r['keywords']}/{n_kw}" for r in rs)
            costs = [r["cost"] for r in rs]
            cost = "算定不可" if any(c is None for c in costs) else spread(costs, "{:.6f}")
            lines.append("| " + " | ".join([
                f"{q['id']} {q['kind']}", METHODS[method], ev_r, ev_c, kw,
                "・".join(str(len(r["cited_not_retrieved"])) for r in rs),
                spread([len(r["calls"]) for r in rs], "{:.1f}"),
                spread([len(r["retrieved"]) for r in rs], "{:.1f}"),
                spread([r["requests"] for r in rs], "{:.1f}"),
                spread([r["input_tokens"] for r in rs]), spread([r["output_tokens"] for r in rs]),
                spread([r["seconds"] for r in rs], "{:.1f}"), cost]) + " |")

    cached = {m: [r.get("cache_read_tokens") or 0 for r in records if r["method"] == m] for m in methods}
    if any(sum(v) for v in cached.values()):
        lines += ["", "- 入力トークンのうちキャッシュから読んだ分（平均）：" + "、".join(
            f"{METHODS[m]} {mean(v):,.0f}" for m, v in cached.items() if v)]

    if "rag" in methods and "hybrid" in methods:
        lines += ["", "## ハイブリッドと基本RAGの差（平均）", ""]
        for q in questions:
            base, new = by.get((q["id"], "rag")), by.get((q["id"], "hybrid"))
            if not base or not new:
                continue
            def avg(rs, key):
                return mean(r[key] for r in rs)
            parts = []
            if not q.get("unanswerable"):
                n = len(q["evidence"])
                parts.append(f"根拠の取得 {avg(base, 'evidence_retrieved'):.1f}→{avg(new, 'evidence_retrieved'):.1f}/{n}")
                parts.append(f"引用 {avg(base, 'evidence_cited'):.1f}→{avg(new, 'evidence_cited'):.1f}/{n}")
            t0, t1 = avg(base, "input_tokens"), avg(new, "input_tokens")
            parts.append(f"入力トークン {t0:,.0f}→{t1:,.0f}（{(t1 - t0) / t0 * 100:+.0f}%）" if t0 else "入力トークン —")
            parts.append(f"検索回数 {mean(len(r['calls']) for r in base):.1f}→{mean(len(r['calls']) for r in new):.1f}")
            parts.append(f"時間 {avg(base, 'seconds'):.1f}→{avg(new, 'seconds'):.1f}秒")
            lines.append(f"- {q['id']}：" + "、".join(parts))

    lines += ["", "## 各回の記録", "",
              "ページは「回-PDFページ」。★は根拠ページ。検索語はツールに渡された引数。", ""]
    for q in questions:
        gold = {p for g in q["evidence"] for p in g}
        lines.append(f"### {q['id']} {q['kind']}：{q['question']}")
        if q["evidence"]:
            lines.append("- 根拠の組：" + " ／ ".join("または".join(label(p) for p in g) for g in q["evidence"]))
        for method in methods:
            for i, r in enumerate(by.get((q["id"], method), []), 1):
                mark = lambda ps: "、".join(("★" if p in gold else "") + label(p) for p in ps) or "なし"
                args = "／".join(json.dumps(c["args"], ensure_ascii=False)[:80] for c in r["calls"]) or "呼ばず"
                lines.append(f"- {METHODS[method]} {i}回目：検索語 {args}")
                lines.append(f"  - 取得：{mark(r['retrieved'])}")
                lines.append(f"  - 引用：{mark(r['cited'])}" + (f"（取得していない：{mark(r['cited_not_retrieved'])}）" if r["cited_not_retrieved"] else ""))
        lines.append("")
    lines += ["## 回答（各方式の1回目）", ""]
    for q in questions:
        for method in methods:
            rs = by.get((q["id"], method), [])
            if rs:
                lines += [f"<details><summary>{q['id']} {METHODS[method]}</summary>", "", rs[0]["answer"].strip(), "", "</details>", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="試作品：3問 × 基本RAG・ハイブリッド × 1回")
    parser.add_argument("--methods", default=None, help="例：rag,graph,tree,hybrid（既定：rag,graph,hybrid）")
    parser.add_argument("--trials", type=int, default=None, help="各方式の実行回数（既定：3、--quickは1）")
    parser.add_argument("--questions", default=None, help="例：Q1,Q4")
    parser.add_argument("--sleep", type=float, default=4.0, help="1回ごとに待つ秒数（レート制限よけ）")
    parser.add_argument("--selfcheck", action="store_true", help="APIを使わず、偽のモデルと保存済みの埋め込みで仕組みだけ確かめる")
    args = parser.parse_args()

    methods = (args.methods or ("rag,hybrid" if args.quick else "rag,graph,hybrid")).split(",")
    assert all(m in METHODS for m in methods), f"方式は {', '.join(METHODS)} から選ぶ"
    trials = args.trials or (1 if args.quick or args.selfcheck else 3)
    ids = (args.questions or (",".join(QUICK) if args.quick else ",".join(q["id"] for q in QUESTIONS))).split(",")
    questions = [q for q in QUESTIONS if q["id"] in ids]
    out = OUT
    if args.selfcheck:
        selfcheck_setup()
        methods, questions, out = list(METHODS), questions[:2], Path(tempfile.mkdtemp())
        model = selfcheck_model()
    else:
        model = A.build_model()
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    records = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if runs_path.exists() else []
    records = [r for r in records if r["model"] == MODEL]
    done = {}
    for r in records:
        done[(r["qid"], r["method"])] = done.get((r["qid"], r["method"]), 0) + 1

    todo = [(q, m) for t in range(trials) for q in questions for m in methods if done.get((q["id"], m), 0) <= t]
    print(f"[model] {MODEL}　実行 {len(todo)} 回（保存済み {len(records)} 回は飛ばす）", flush=True)
    for i, (q, method) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {q['id']} {METHODS[method]}", flush=True)
        try:
            record = run_with_retry(method, q, model)
        except RetryLater:
            print("レート制限が続いています。時間をおいて同じコマンドを再実行すると、続きから再開します。")
            break
        except ModelHTTPError as e:
            print(api_error_message(e.status_code, str(e.body)[:200]))
            print("時間をおいて同じコマンドを再実行すると、続きから再開します。")
            break
        with runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        print(f"    根拠 取得{record['evidence_retrieved']}・引用{record['evidence_cited']}／{len(q['evidence'])}"
              f"　入力{record['input_tokens']:,}　{record['seconds']}秒", flush=True)
        if not args.selfcheck and i < len(todo):
            time.sleep(args.sleep)

    keep = [r for r in records if r["qid"] in ids and r["method"] in methods]
    (out / "summary.md").write_text(summarize(keep, questions, methods, trials), encoding="utf-8")
    print(f"\n集計：{(out / 'summary.md').relative_to(out.parent.parent.parent) if not args.selfcheck else out / 'summary.md'}")
    if args.selfcheck:
        assert len(keep) == len(questions) * len(methods)
        assert cited_pages("第3回・PDF p.13、39、40（第2回・PDF p.52〜53）") == [
            "lec03-main-p013", "lec03-main-p039", "lec03-main-p040", "lec02-main-p052", "lec02-main-p053"]
        print("selfcheck ok")


def selfcheck_setup() -> None:
    """APIを呼ばないよう、質問の埋め込みを保存済みのページ埋め込みで置き換える。"""
    import struct
    from app import hybrid, retrieval
    from app.config import RAG_DB
    with retrieval.open_index(RAG_DB) as db:
        row = db.execute("SELECT embedding FROM chunks WHERE page_id = 'lec02-main-p052'").fetchone()
    vector = list(struct.unpack(f"{len(row[0]) // 4}f", row[0]))
    retrieval.embed_text = hybrid.embed_text = lambda text, task: vector


def selfcheck_model():
    from pydantic_ai.models.test import TestModel
    return TestModel()


if __name__ == "__main__":
    main()
