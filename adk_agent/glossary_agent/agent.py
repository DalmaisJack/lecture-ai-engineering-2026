"""第3回 宿題（発展 a）：演習のエージェントを Google ADK で組み立て直す。

指示（INSTRUCTIONS）とツール（app/tools.py）は演習のものをそのまま使い、
エージェントを組み立てる部分だけを ADK で書く。app/ のファイルは変更しない。

使い方（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk==2.13.0 adk run --in_memory adk_agent/glossary_agent "質問"
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

# 演習で使ったツール。make_quiz を作った場合だけ、それも渡す
TOOLS = [tools.search_glossary, tools.get_lecture_summary]
if hasattr(tools, "make_quiz"):
    TOOLS.append(tools.make_quiz)


# --- --trace の代わりの表示（ふるまいは変えない。None を返すと何も上書きしない） ---
# ADK はキーワード引数で呼ぶので、引数名は変えないこと
def show_tool_call(tool, args, tool_context):
    print(f"[tool call]   {tool.name}({args})")
    return None


def show_tool_return(tool, args, tool_context, tool_response):
    print(f"[tool return] {tool.name} -> {str(tool_response)[:120]}…")
    return None


def show_llm_call(callback_context, llm_response):
    usage = llm_response.usage_metadata
    if usage and not llm_response.partial:
        print(f"[llm call]    input={usage.prompt_token_count} output={usage.candidates_token_count}")
    return None


root_agent = Agent(
    model=MODEL,  # Pydantic AI 版と同じモデル（既定は gemini-3.5-flash-lite）
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=TOOLS,
    before_tool_callback=show_tool_call,
    after_tool_callback=show_tool_return,
    after_model_callback=show_llm_call,
)
