"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


def chat(first=None):
    """第1回 宿題（実装）：会話履歴を保ったまま続けて質問する"""
    import sys
    from google.genai import types

    client = genai.Client(api_key=get_api_key())
    history = []    # これまでの user / model の発言。毎回まとめて送る
    max_turns = 10  # 送る履歴は直近10往復まで（トークンの増えすぎ防止）
    print(f"[会話モード] モデル: {MODEL}（exit か 終了 で終わる）")
    while True:
        if first:
            q, first = first, None
            print(f"\nあなた> {q}")
        else:
            try:
                q = input("\nあなた> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not sys.stdin.isatty():
                print(q)  # パイプで質問を流したときは記録用に表示
        if q in ("exit", "quit", "終了"):
            break
        if not q:
            continue
        contents = history + [types.Content(role="user", parts=[types.Part(text=q)])]
        try:
            res = client.models.generate_content(model=MODEL, contents=contents)
        except Exception as e:
            print(f"[エラー] {e}（この質問は履歴に入れない）")
            continue
        reply = res.candidates[0].content if res.candidates else None
        if not reply or not reply.parts:
            print("[応答が空でした]")
            continue
        print(f"\nGemini> {res.text}")
        u = res.usage_metadata
        if u:
            print(f"[入力 {u.prompt_token_count} / 出力 {u.candidates_token_count} トークン]")
        history = (contents + [reply])[-2 * max_turns:]


if __name__ == "__main__":
    import sys  # 第1回 宿題（実装）: --chat で会話モード
    if sys.argv[1:2] == ["--chat"]:
        chat(" ".join(sys.argv[2:]) or None)
        raise SystemExit(0)
    main()
