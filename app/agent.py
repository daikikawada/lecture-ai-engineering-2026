"""第3回: Pydantic AI で作る学習支援エージェント。

使い方:
    uv run python -m app.agent "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
モデルは tools.py の関数を「ツール」として呼び出し、その結果を見て回答を作る。
第5回では、同じツールを MCP サーバー（app/mcp_server.py）経由で受け取る。
"""

import argparse
import inspect
import json
from pathlib import Path

from pydantic_ai import (
    Agent,
    AgentRunResult,
    DeferredToolRequests,
    DeferredToolResults,
)
from pydantic_ai.exceptions import ModelHTTPError
from fastmcp import Client
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.google import GoogleProvider

from app import tools
from app.config import MODEL, api_error_message, get_api_key

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 学習記録の保存を依頼されたら save_learning_log を使う。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

# モデルは実行時に渡す（テストではAPIを呼ばない TestModel に差し替えるため）
agent = Agent(output_type=[str, DeferredToolRequests], instructions=INSTRUCTIONS)
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)
agent.tool_plain(tools.make_quiz)
agent.tool_plain(tools.save_learning_log, requires_approval=True)

# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[MCPToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（MCPToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。

    Pydantic AI には同じことをする load_mcp_toolsets() もある。ここで自分で組み立てているのは、
    古い SDK で作られたサーバー（fetch など）につなぐとき、接続方式の自動判定（"auto"）が
    警告を大量に出すため。"legacy" に固定すると新旧どちらのサーバーにも静かにつながる。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            transport = spec["url"]
        else:
            transport = StdioTransport(spec["command"], spec.get("args", []), env=spec.get("env"))
        toolsets.append(MCPToolset(Client(transport, mode="legacy")).prefixed(name))
    return toolsets


def build_mcp_agent(config_path: Path = MCP_CONFIG) -> Agent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。"""
    return Agent(instructions=INSTRUCTIONS, toolsets=load_mcp_servers(config_path))


def build_model() -> GoogleModel:
    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key()))


def print_tools(target: Agent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    TestModel（テストで使う偽モデル）にツール一覧を渡し、そこから読み出す。
    MCP サーバーのツールも、第3回の関数ツールも、モデルから見れば同じ形をしている。
    """
    model = TestModel(call_tools=[])
    target.run_sync("ツール一覧", model=model)
    tools_seen = model.last_model_request_parameters.function_tools
    print(f"モデルに見えているツール: {len(tools_seen)} 個")
    for tool in tools_seen:
        print(f"\n[{tool.name}]")
        print("  " + inspect.cleandoc(tool.description or "").replace("\n", "\n  "))
        params = tool.parameters_json_schema.get("properties", {})
        print(f"  引数: {json.dumps(params, ensure_ascii=False)}")


def print_trace(result) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。"""
    print("\n--- trace ---")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"[tool return] {part.tool_name} -> {str(part.content)[:120]}…")
    print(f"[usage] {result.usage}")


def confirm_tool_approvals(
    requests: DeferredToolRequests,
) -> DeferredToolResults:
    """承認待ちのツール呼び出しについて利用者に確認する。

    Args:
        requests: 承認待ちとなったツール呼び出し情報。

    Returns:
        各ツール呼び出しの承認・拒否を格納したDeferredToolResults。

    処理:
        1. requests.approvals に含まれる各ツールを確認する。
        2. ツール名、引数、保存予定の学習記録をCLIへ表示する。
        3. 利用者に y/N で承認を確認する。
        4. y の場合のみ承認し、それ以外は拒否する。
        5. tool_call_id をキーとした承認結果を返す。
    """
    approvals: dict[str, bool] = {}
    for tool_call in requests.approvals:
        print(f"\n承認待ちツール: {tool_call.tool_name}")
        args = tool_call.args
        if isinstance(args, dict):
            print(f"引数: {json.dumps(args, ensure_ascii=False)}")
            if tool_call.tool_name == "save_learning_log" and "content" in args:
                print("保存予定の学習記録:")
                print(args["content"])
        else:
            print(f"引数: {args}")

        approved = (
            input("このツール呼び出しを承認しますか？ [y/N]: ")
            .strip()
            == "y"
        )
        approvals[tool_call.tool_call_id] = approved

    return DeferredToolResults(approvals=approvals)


def run_with_approval(
    target: Agent[None, str | DeferredToolRequests],
    question: str,
) -> AgentRunResult[str | DeferredToolRequests]:
    """利用者の承認を必要とするツールを含むエージェントを実行する。

    Args:
        target: 実行対象のエージェント。
        question: 利用者からの質問または指示。

    Returns:
        承認処理を完了した後の最終的なエージェント実行結果。

    処理:
        1. エージェントを実行する。
        2. 出力がDeferredToolRequestsの場合は処理を一時停止する。
        3. confirm_tool_approvalsで利用者の承認を取得する。
        4. message_historyとdeferred_tool_resultsを渡して再開する。
        5. 承認待ちが解消されるまで処理を繰り返す。
    """
    result = target.run_sync(question)
    while isinstance(result.output, DeferredToolRequests):
        deferred_tool_results = confirm_tool_approvals(result.output)
        result = target.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=deferred_tool_results,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument("--mcp", action="store_true", help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る")
    parser.add_argument("--list-tools", action="store_true", help="モデルに見えるツール一覧を表示して終了する（APIキー不要）")
    args = parser.parse_args()

    target = build_mcp_agent() if args.mcp else agent
    if args.list_tools:
        print_tools(target)
        return
    if not args.question:
        parser.error("質問を指定してください")

    print(f"[model] {MODEL}")
    if args.mcp:
        print("[tools] MCP サーバー（app/mcp_servers.json）")
    try:
        if args.mcp:
            result = target.run_sync(args.question, model=build_model())
        else:
            with target.override(model=build_model()):
                result = run_with_approval(target, args.question)
    except ModelHTTPError as e:
        print(api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)
    print(result.output)
    if args.trace:
        print_trace(result)


if __name__ == "__main__":
    main()
