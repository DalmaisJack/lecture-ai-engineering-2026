"""第5回 宿題（発展）：lectures/05/hw の実行結果から、answer.md の「宿題」欄を書き直す。

    uv run python -m scripts.write_homework

数字・ページ・検索語は runs.jsonl と summary.md から入れる。手で直したいところは、実行後に answer.md を直す。
"""

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import MODEL
from app.hybrid import lectures_in
from app.retrieval import load_pages
from scripts.compare_retrieval import METHODS, QUESTIONS

ROOT = Path(__file__).resolve().parents[1]
HW = ROOT / "lectures/05/hw"
ANSWER = ROOT / "lectures/05/answer.md"
PAGES = {p["page_id"]: p for p in load_pages()}
Q = {q["id"]: q for q in QUESTIONS}
NAMES = {  # 根拠の組の呼び方（QUESTIONS の evidence と同じ順）
    "Q1": ["LoRAとQLoRA"],
    "Q2": ["メモリの考え方", "コンパクション", "日々のメモと長期メモリ"],
    "Q3": ["長い文脈の性質", "エージェントで起きること", "メモリでの対処"],
    "Q4": ["LoRA", "メモリ"],
}


def num(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def page(pid: str, title: bool = False) -> str:
    p = PAGES[pid]
    return f"第{p['lecture']}回 p.{p['page']}" + (f"「{p['title'][:18]}」" if title else "")


def where(group: list[str]) -> str:
    """根拠の組を「第3回 p.38〜40 のどれか」の形にする。"""
    by = {}
    for pid in group:
        by.setdefault(PAGES[pid]["lecture"], []).append(PAGES[pid]["page"])
    parts = []
    for lecture, ps in by.items():
        ps = sorted(ps)
        if len(ps) >= 3 and ps[-1] - ps[0] == len(ps) - 1:
            parts.append(f"第{lecture}回 p.{ps[0]}〜{ps[-1]}")
        else:
            parts.append(f"第{lecture}回 " + "・".join(f"p.{n}" for n in ps))
    return "、".join(parts) + ("" if len(group) == 1 else " のどれか")


def short(text: str, n: int = 28) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


runs = [json.loads(line) for line in (HW / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
runs = [r for r in runs if r["model"] == MODEL]
by = {}
for r in runs:
    by.setdefault((r["qid"], r["method"]), []).append(r)
qids = [q["id"] for q in QUESTIONS if any((q["id"], m) in by for m in METHODS)]
methods = [m for m in METHODS if any((qid, m) in by for qid in qids)]
counts = [len(by[(qid, m)]) for qid in qids for m in methods if (qid, m) in by]
times = f"{min(counts)}回ずつ" if min(counts) == max(counts) else f"{min(counts)}〜{max(counts)}回"


def stats(qid: str, m: str) -> dict | None:
    rs = by.get((qid, m))
    if not rs:
        return None
    return {"ev_r": mean(r["evidence_retrieved"] for r in rs), "ev_c": mean(r["evidence_cited"] for r in rs),
            "tok": mean(r["input_tokens"] for r in rs), "sec": mean(r["seconds"] for r in rs),
            "calls": mean(len(r["calls"]) for r in rs), "runs": rs,
            "retrieved": {p for r in rs for p in r["retrieved"]}, "cited": {p for r in rs for p in r["cited"]}}


pairs = {qid: (stats(qid, "rag"), stats(qid, "hybrid")) for qid in qids if Q[qid]["evidence"]}
pairs = {qid: p for qid, p in pairs.items() if p[0] and p[1]}


def change(qid: str) -> tuple:
    a, b = pairs[qid]
    return (b["ev_r"] - a["ev_r"], b["ev_c"] - a["ev_c"], (a["tok"] - b["tok"]) / a["tok"])


def facts(qid: str) -> str:
    a, b = pairs[qid]
    n = len(Q[qid]["evidence"])
    if a["ev_r"] == b["ev_r"] and a["ev_c"] == b["ev_c"]:
        head = f"根拠は基本RAGと同じ（取得{num(b['ev_r'])}/{n}・引用{num(b['ev_c'])}/{n}）のまま"
    else:
        head = f"根拠の取得が{num(a['ev_r'])}→{num(b['ev_r'])}/{n}、引用が{num(a['ev_c'])}→{num(b['ev_c'])}/{n}になり"
    pct = (b["tok"] - a["tok"]) / a["tok"] * 100
    return (f"{qid}・{Q[qid]['kind']}。{head}、入力トークンが{a['tok']:,.0f}→{b['tok']:,.0f}（{pct:+.0f}%）、"
            f"時間が{a['sec']:.1f}→{b['sec']:.1f}秒になった。検索回数は基本RAGが{num(a['calls'])}回、ハイブリッドが{num(b['calls'])}回だった。")


lines_good, lines_bad, results_extra = [], [], []
DISPLAY = {"rag": "基本RAG", "graph": "元のGraph", "tree": "元のTree", "hybrid": "ハイブリッド"}
ranked = sorted(pairs, key=change, reverse=True)
by_evidence = [q for q in ranked if change(q)[0] > 0 or change(q)[1] > 0]
by_tokens = [q for q in ranked if change(q)[0] >= 0 and change(q)[1] >= 0 and change(q)[2] > 0.1]
picked = list(dict.fromkeys(by_evidence[:1] + sorted(by_tokens, key=lambda q: change(q)[2], reverse=True)[:1]))
for best in picked:
    a, b = pairs[best]
    text = facts(best)
    lectures = sorted({PAGES[p]["lecture"] for p in b["retrieved"] if p in PAGES})
    if a["calls"] > b["calls"] and len(lectures) >= 2:
        first = b["runs"][0]["calls"][0]["args"].get("query", "") if b["runs"][0]["calls"] else ""
        how = "質問文のままの1回目の検索" if " ".join(first.split()) == " ".join(Q[best]["question"].split()) else f"1回目の検索（「{short(first, 20)}」）"
        text += (f"ハイブリッドは{how}で第{'回と第'.join(map(str, lectures))}回のページがそろい、"
                 + ("検索し直さずに答えた。" if b["calls"] == 1 else "検索し直す回数が減った。"))
        queries = [c["args"].get("query", "") for c in a["runs"][0]["calls"]]
        if len(queries) >= 2:
            text += "基本RAGは" + "".join(f"「{short(q, 20)}」" for q in queries[:3]) + "のように分けて検索していた。"
    gained = sorted({p for r in b["runs"] for p in r["retrieved"]} - a["retrieved"],
                    key=lambda p: (PAGES[p]["lecture"], PAGES[p]["page"]))
    gold = [p for g in Q[best]["evidence"] for p in g]
    new_gold = [p for p in gained if p in gold]
    if change(best)[0] > 0 and new_gold:
        text += "基本RAGが取れなかった根拠の" + "、".join(page(p, title=True) for p in new_gold) + "が、ハイブリッドでは入った。"
    lines_good.append(text)
if not picked:
    lines_good.append("根拠が増えた質問も、根拠が同じでトークン数が1割以上減った質問もなかった。")

worst = ranked[-1] if ranked else None
if worst and worst not in picked:
    w = change(worst)
    if w[0] < 0 or w[1] < 0 or w[2] < -0.05:
        a, b = pairs[worst]
        text = facts(worst)
        extra = sorted(b["retrieved"] - a["retrieved"], key=lambda p: (PAGES[p]["lecture"], PAGES[p]["page"]))
        if extra:
            used = [p for p in extra if p in b["cited"]]
            text += "ハイブリッドだけが取得したのは" + "、".join(page(p, title=True) for p in extra[:4])
            text += (("で、回答では引用されなかった。" if len(extra) == 1 else "で、回答ではどれも引用されなかった。") if not used else
                     f"で、回答で引用されたのは{'、'.join(page(p) for p in used)}だけだった。")
        else:
            text += "取得したページは基本RAGと同じなので、増えたのは指示文・ツールの説明・選んだ理由（why）の分だと思う。"
        lines_bad.append(text)
for qid, (a, b) in pairs.items():
    if qid not in picked and any(len(r["calls"]) >= 2 for r in b["runs"]):
        r = next(r for r in b["runs"] if len(r["calls"]) >= 2)
        lines_bad.append(f"{qid}ではハイブリッドも検索し直した回があった（2回目の検索語は「{short(r['calls'][1]['args'].get('query', ''))}」）。"
                         "指示文には、足りない話題があるときだけ検索し直すと書いている。")
        break
for qid, (a, b) in pairs.items():
    for i, group in enumerate(Q[qid]["evidence"]):
        missed = sum(not set(group) & set(r["retrieved"]) for r in b["runs"])
        if missed:
            also = sum(not set(group) & set(r["retrieved"]) for r in a["runs"])
            name = NAMES.get(qid, [""] * 9)[i]
            one = len(b["runs"]) == 1
            lines_bad.append(f"{qid}で、ハイブリッドは「{name}」の根拠（{where(group)}）を"
                             + ("取れなかった" if one else f"{missed}/{len(b['runs'])}回取れなかった")
                             + (("（基本RAGも取れなかった）。" if one else f"（基本RAGも{also}/{len(a['runs'])}回取れなかった）。") if also else "（基本RAGは取れていた）。"))

for qid in [qid for qid in qids if Q[qid].get("unanswerable")]:
    told = {m: (sum(bool(r["declined"]) for r in by[(qid, m)]), len(by[(qid, m)])) for m in methods if (qid, m) in by}
    results_extra.append(f"資料にない質問（{qid}）には、" + "、".join(f"{DISPLAY[m]}が{k}/{n}回" for m, (k, n) in told.items())
                         + "「確認できない」と答えた")
    if "hybrid" in told and told["hybrid"][0] < told["hybrid"][1]:
        lines_bad.append(f"資料にない質問（{qid}）で、ハイブリッドが「確認できない」と答えなかった回があった（{told['hybrid'][1] - told['hybrid'][0]}/{told['hybrid'][1]}回）。")

summary = (HW / "summary.md").read_text(encoding="utf-8").splitlines()
table = [line for line in summary if line.startswith("|")]
legend = [line for line in summary if line.startswith(("- 根拠の取得・引用", "- 要点の語句", "- 数値は平均"))]
legend.append("- 資料にない質問は、回答に「確認できない」「記載されておりません」などの言い回しがあるかで判定した（機械的な目安）")
diffs, inside = [], False
for line in summary:
    if line.startswith("## "):
        inside = line.startswith("## ハイブリッドと基本RAGの差")
    elif inside and line.startswith("- "):
        diffs.append(line)
cache_line = [line for line in summary if line.startswith("- 入力トークンのうちキャッシュ")]

graph_line = ""
if "graph" in methods:
    with_ev = [qid for qid in qids if Q[qid]["evidence"] and (qid, "graph") in by and (qid, "hybrid") in by]
    if with_ev:
        rate = lambda qid, m: mean(r["evidence_retrieved"] for r in by[(qid, m)]) / len(Q[qid]["evidence"])
        tok = lambda qid, m: mean(r["input_tokens"] for r in by[(qid, m)])
        graph_line = (f"元のGraphと比べると、根拠の取得率は平均{mean(rate(q, 'graph') for q in with_ev):.0%}→"
                      f"{mean(rate(q, 'hybrid') for q in with_ev):.0%}、入力トークンは平均"
                      f"{mean(tok(q, 'graph') for q in with_ev):,.0f}→{mean(tok(q, 'hybrid') for q in with_ev):,.0f}だった。")

trace = HW / "trace_hybrid_q4.txt"
trace_lines = []
if trace.exists():
    body = trace.read_text(encoding="utf-8").splitlines()
    if "--- trace ---" in body:
        trace_lines = [l for l in body[body.index("--- trace ---") + 1:] if l.startswith("[") or l.startswith("    - ")]

multi = [qid for qid in pairs if len({PAGES[p]["lecture"] for g in Q[qid]["evidence"] for p in g}) >= 2]
notes = []
if len(multi) >= 2:
    pcts = {qid: (pairs[qid][1]["tok"] - pairs[qid][0]["tok"]) / pairs[qid][0]["tok"] * 100 for qid in multi}
    same_sign = len({p < 0 for p in pcts.values()}) == 1
    notes.append("回をまたぐ質問では、入力トークンが" + "、".join(f"{qid}で{p:+.0f}%" for qid, p in pcts.items())
                 + ("と、同じ向きに動いた。" if same_sign else "と、同じ種類の質問でも結果が分かれた。"))
    fewer = {qid: pairs[qid][0]["calls"] > pairs[qid][1]["calls"] for qid in multi}
    if all(fewer[qid] == (pcts[qid] < 0) for qid in multi):
        notes.append("検索回数（基本RAG→ハイブリッド）は" + "、".join(
            f"{qid}が{num(pairs[qid][0]['calls'])}→{num(pairs[qid][1]['calls'])}回" for qid in multi)
            + "で、検索のやり直しが減ったときにトークン数が減っている。入口を足す効き目は、ページが増えることより、検索し直さずに済むことの方が大きいと思う。")
    else:
        notes.append("検索回数の差だけでは、トークン数の差を説明できなかった。")
common = [q for q in pairs if (q, "graph") in by]
if common:
    r_tok = mean(pairs[q][0]["tok"] for q in common)
    g_tok = mean(mean(r["input_tokens"] for r in by[(q, "graph")]) for q in common)
    h_tok = mean(pairs[q][1]["tok"] for q in common)
    notes.append(f"元のGraphは入力トークンが基本RAGの平均{g_tok / r_tok:.1f}倍になったのに対し、ハイブリッドは{h_tok / r_tok:.1f}倍だった。"
                 + ("足すページを質問との近さで絞り、3件までにしたことが効いていると思う。" if h_tok < g_tok else
                    "ハイブリッドにしても、元のGraphより入力トークンは減らなかった。"))
asked_lec = [r for (qid, m), rs in by.items() if m == "hybrid" for r in rs
             if r["calls"] and lectures_in(r["question"], {1, 2, 3})]
if asked_lec:
    same = sum(" ".join(r["calls"][0]["args"].get("query", "").split()) == " ".join(r["question"].split()) for r in asked_lec)
    kept = sum(set(lectures_in(r["question"], {1, 2, 3})) <= set(lectures_in(r["calls"][0]["args"].get("query", ""), {1, 2, 3}))
               for r in asked_lec) - same
    n = len(asked_lec)
    notes.append(f"ハイブリッドの1回目の検索語は、{n}回とも質問文そのままだった。" if same == n else
                 f"ハイブリッドの1回目の検索語が質問文そのままだったのは{same}/{n}回で、"
                 + ("ほかは「第N回」を残して短くしていた。" if kept == n - same else f"ほかは短くしていた（「第N回」を残したのは{kept}回）。")
                 + "指示文には質問文をそのまま入れると書いたが、短くしても回の番号が残っていれば範囲の絞り込みは効く。")
if max(counts) == 1:
    notes.append("各1回の結果なので偶然の差もありそうで、次は回数を増やして確かめたい。" if "graph" in methods else
                 "各1回の結果なので偶然の差もありそうで、次は回数を増やし、元のGraphとも比べたい。")
elif picked:
    q0 = picked[0]
    a, b = pairs[q0]
    if change(q0)[0] > 0:
        key, what = "evidence_retrieved", "ハイブリッドの方が根拠を多く取れ"
        wins = sum(y[key] > x[key] for x, y in zip(a["runs"], b["runs"]))
    elif change(q0)[1] > 0:
        key, what = "evidence_cited", "ハイブリッドの方が引用できた根拠が多く"
        wins = sum(y[key] > x[key] for x, y in zip(a["runs"], b["runs"]))
    else:
        what = "ハイブリッドの方が入力トークンが少なく"
        wins = sum(y["input_tokens"] < x["input_tokens"] for x, y in zip(a["runs"], b["runs"]))
    n = min(len(a["runs"]), len(b["runs"]))
    notes.append(f"{q0}の結果はまだ{n}回分なので、偶然の差かもしれない。回数を増やして確かめたい。" if n < 2 else
                 f"{q0}は{wins}/{n}回で{what}、1回だけの偶然ではなかった。" if wins == n else
                 f"{q0}で{what.replace('少なく', '少なかった').replace('取れ', '取れた').replace('多く', '多かった')}のは{wins}/{n}回で、回によって結果がぶれた。")
good_q = set(picked)
single = [q for q in pairs if len({PAGES[p]["lecture"] for g in Q[q]["evidence"] for p in g}) == 1 and change(q)[2] < -0.05 and change(q)[0] <= 0]
if good_q and single:
    notes.append("講義の質問に答える用途で使うなら、質問に2つ以上の回や用語が出てくるときだけハイブリッドに切り替えたい。"
                 f"1つの回で答えられる質問（{'・'.join(single)}）では、ページや説明が増える分だけ損になるので、基本RAGのままにする。")
elif good_q:
    notes.append("講義の質問に答える用途なら、ハイブリッドを標準にしてもよいと思う。")
else:
    notes.append("今回の結果だけでは、基本RAGから切り替える理由はまだない。")
if any(len(r["calls"]) >= 2 for qid in pairs for r in pairs[qid][1]["runs"]):
    notes.append("疑問：指示文に検索し直す条件を書いても、ハイブリッドで検索し直す場合とそうでない場合があった。モデルが何で決めているのかが気になった。")
else:
    notes.append("疑問：用語集にない言葉（コンパクションなど）からも入口を作るには、どうすればよいか。")

asked = "\n".join(f"  - {qid}・{Q[qid]['kind']}：「{Q[qid]['question'].replace('「', '『').replace('」', '』')}」"
                  + (" 根拠：" + "／".join(where(g) for g in Q[qid]["evidence"]) if Q[qid]["evidence"] else " 根拠：資料にない（確認できないと答えるのが正解）")
                  for qid in qids)
method_names = "、".join({"rag": "基本RAG（`search_materials`）", "graph": "元のGraph（`search_graph`）",
                          "tree": "元のTree（`browse_tree`）", "hybrid": "ハイブリッド（`search_hybrid`）"}[m] for m in methods)

section = f"""- 取り組んだレベル：発展
- やったこと（変更したファイル・設定・処理、または設計した取得方法）：
  - 解きたい質問：回をまたぐ説明と、回をまたぐ用語の比較。基本RAGは質問全体に近い順に5件を返すので、話題が2つあると片方の回に寄って、もう片方の根拠が入らないことがあると思った
  - 取得方法：ベクトル検索に、Tree（回・章）と Graph（用語・前後ページ）を組み合わせた「ハイブリッド」を作り、`--hybrid` で切り替えられるようにした
    1. 範囲：質問に「第N回」があれば、その回のページだけを候補にする
    2. 土台：範囲の中で質問に近い上位5件（基本RAGと同じ件数）
    3. 足す（合計3件まで）：質問に2つ以上の回が出てくるときは、回ごとの入口（その回で質問に最も近いページ）。質問に出てくる用語ごとの入口（その用語に言及するページのうち、見出しにその用語があるものを優先して、質問に最も近いページ。言及ページが少ない用語から2件まで）。1位のページと同じ章の前後ページ（範囲の中で質問に近い順15位以内のときだけ）。本文150字未満の区切りのスライドは足さない
    4. ページごとに、選んだ理由（why）を付けて返す
  - データの形・持ち方：新しいデータは作っていない。本文は `lectures.json`、ページの埋め込みは同梱の `index.sqlite3`、回と章は教材の階層、用語とページの関係（mentions）と前後ページ（next_page）は `graph.json` をそのまま使う。質問の埋め込みは1回の検索につき1回で、索引は作り直していない
  - 選んだ理由：
    - 質問に書かれた回と用語は、答えがある場所の手がかりになる。回は目次の上の階層、用語は graph.json の用語ノードに対応するので、話題ごとに入口を1つずつ作れる
    - 元の Graph は、入口から前後ページと共通用語へ決まった順で広げ、最大10件を返す。質問との近さを見ずに広げるので、関係の薄いページも入りうる。なので足すページは質問との近さで絞り、3件までにした
    - 用語どうしの関係（related_term）は使わなかった。LoRA からたどると、言及ページが39あるファインチューニングや10ある Hugging Face Hub に広がる。メモリや RAG にはこの関係がないので、今回の質問の片方では入口にならない
  - 変更したファイル：`app/hybrid.py`（取得処理）、`app/tools.py`（ツール `search_hybrid`）、`app/agent.py`（指示文 `HYBRID_INSTRUCTIONS`、`--hybrid`、trace の表示）、`tests/test_hybrid.py`（APIを使わないテスト）、`scripts/compare_retrieval.py`（比較の実行と集計）、`scripts/write_homework.py`（この欄の数字の書き込み）
- 比較した質問と、固定した条件（モデル・対象資料など）：
{asked}
  - 根拠のページは、実行の前に教材JSONの本文を読んで決めた
  - 固定した条件：モデルは {MODEL}、対象資料は第1〜3回の本講義214ページ、検索DBは同梱のもの。{method_names}で、各質問を{times} `scripts/compare_retrieval.py` で実行した。ハイブリッドの指示文は、検索ツールの使い方に関わる3か所だけ基本RAGと違う。サーバー側のエラー（503・500）が出た回は30〜60秒待ってやり直し、時間はやり直した1回だけを測った
- 結果（変更前後の回答・根拠、`--trace`の出力、時間・トークン数・生成API推定費用など）：

{chr(10).join(legend)}

{chr(10).join(table)}

{chr(10).join(diffs + cache_line)}
""" + "".join(f"- {t}\n" for t in results_extra) + (f"- {graph_line}\n" if graph_line else "") + (
    "- 改良版の `--trace`（Q4、1回）：\n\n```text\n" + "\n".join(trace_lines) + "\n```\n" if trace_lines else "") + f"""- 回答の全文、ツールに渡した検索語、取得・引用したページは `lectures/05/hw/summary.md` と `runs.jsonl` に残した
- 改善した例と、残る失敗・制約：
""" + "".join(f"  - 改善した例：{t}\n" for t in lines_good) + "".join(
    f"  - {'失敗した例' if i == 0 else '残る失敗'}：{t}\n" for i, t in enumerate(lines_bad)) + f"""  - 制約：用語集にない言葉（コンパクションなど）は入口にならない。サーバー側のエラーで待ったあとの実行もあり、時間はぶれやすい
- 気づき・疑問：{''.join(notes)}
"""

text = ANSWER.read_text(encoding="utf-8")
start = text.find("## 宿題")
if start < 0:
    head = text.rstrip("\n") + "\n\n## 宿題\n\n"
else:  # 見出しと案内のコメントは元のまま残し、「- 取り組んだレベル」から下を書き直す
    level = text.find("- 取り組んだレベル", start)
    head = text[:level] if level >= 0 else text[:start] + "## 宿題\n\n"
ANSWER.write_text(head + section, encoding="utf-8")
print(f"answer.md の宿題欄を書き直しました（{len(runs)}回分の結果から）")
