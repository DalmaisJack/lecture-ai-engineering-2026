# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
- `--trace` で見えたツール呼び出し（ツール名と引数）：
- 1つの質問での LLM 呼び出し回数（`requests`）：
- 追加したツールと、docstring に書いた説明：

## 宿題

- 取り組んだレベル：発展（a）ADK で作り直す
- やったこと：
  - `adk_agent/glossary_agent/` に `agent.py` と `__init__.py` を作成。INSTRUCTIONS とツールは `app/` から import し、`app/` は変えていない
  - モデルは `app/config.py` の `MODEL`（gemini-3.5-flash-lite）を使い、Pydantic AI 版とそろえた
  - `--trace` の代わりに callback でツール呼び出しと LLM 呼び出しを表示（ふるまいは変えない）
  - `google-adk==2.13.0` に固定し、`adk run --in_memory` で1問ずつ実行（各1回）
- 結果：

| 質問 | Pydantic AI：ツール / requests / 入力・出力トークン | ADK：ツール / LLM呼び出し / 入力・出力トークン | 回答の違い |
|---|---|---|---|
| LoRAとQLoRAの違いは？ | ツール2回 / 2 / 1196・181 | search_glossary（LoRA、QLoRA）/ 2 / 1290・207 | ADK は回答の最後に「第2回」を示さなかった |
| 3レベルとは？ | search_glossary（3レベル）/ 2 / 783・129 | 同じ / 2 / 879・133 | どちらも用語集どおりに答え、「第1回」を示した |
| 知識蒸留とは？（用語集にない） | search_glossary（知識蒸留）→ 見つからず / 2 / 680・95 | 同じ / 2 / 776・40 | Pydantic AI は一般的な説明と「関係する回：第10回」を付けたが、第10回はツールの結果にも data/ にも無い。ADK は「用語集にはない」とだけ答えた |
| 第2回の確認問題 | get_lecture_summary（2）/ 2 / 1180・169 | 同じ / 2 / 1276・246 | Pydantic AI は記述式で、問題の直後に答え（LoRA）を書いていた。ADK は3択で答えは書かなかった。どちらも第2回を示した |

- 書き方の違い：
  - 組み立て：Pydantic AI は `Agent(instructions=…)` を作ってから `agent.tool_plain()` で1つずつ登録し、モデルは実行時に渡す。ADK は `model`・`name`・`instruction`・`tools` をまとめて渡し、`name` が必須
  - ツール：どちらも docstring と型ヒントがそのまま説明文になる
  - 実行：Pydantic AI 版は CLI と APIキーの受け渡しを自分で書いている。ADK は `adk run` が対話とセッション管理を持ち、キーは環境変数から自動で読む
  - トレース：Pydantic AI は実行後にメッセージ履歴を読む。ADK は実行中に callback で割り込む
  - システムプロンプト：ADK は INSTRUCTIONS の後ろに `You are an agent. Your internal name is "glossary_agent".` を自動で足していた（`adk run -v` のログで確認）
- 気づき・疑問：
  - ツールの選び方・引数・回数と LLM 呼び出し回数は4問とも同じだった。指示とツールの説明文が同じなら、フレームワークを替えてもモデルの判断は変わらなかった
  - 入力トークンは ADK が毎回 94〜96 多く、質問や検索結果の長さによらずほぼ一定だった。毎回送る固定部分の差で、ADK が足すシステムプロンプトの1文がその一部だと考えられる
  - 回答の中身は揺れた。ただ各1回の実行なので、フレームワークの差か偶然かは区別できない。何回か繰り返して比べるとどうなるかが疑問として残った
