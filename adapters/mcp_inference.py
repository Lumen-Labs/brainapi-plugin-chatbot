import os
from typing import Any

from src.core.agents.core.parsing import normalize_tool_name
from src.services.mcp.executor import execute_mcp_tool
from src.services.mcp.prompt import build_mcp_tools_instructions
from src.utils.cleanup import strip_json

from adapters.inference import run_inference, stream_inference_events

MAX_MCP_TOOL_ITERATIONS = int(os.getenv("CHATBOT_MCP_TOOL_MAX_ITERATIONS", "5"))


def prepend_mcp_tools_instructions(prompt: str, brain_id: str) -> str:
    instructions = build_mcp_tools_instructions(brain_id)
    return f"{instructions}\n\n{prompt}"


def _parse_tool_call(content: str) -> tuple[str | None, Any]:
    parsed = strip_json(content or "")
    if parsed.get("tool_name") is not None:
        return normalize_tool_name(parsed.get("tool_name")), parsed.get("tool_input")
    return None, None


async def run_inference_with_mcp_tools(
    *,
    provider: str,
    llm_name: str,
    prompt: str,
    brain_id: str,
    brain_pat: str | None,
    max_tokens: int | None = None,
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "user", "content": prepend_mcp_tools_instructions(prompt, brain_id)}
    ]
    for _ in range(MAX_MCP_TOOL_ITERATIONS):
        output = run_inference(
            provider=provider,
            llm_name=llm_name,
            prompt="",
            messages=messages,
            max_tokens=max_tokens,
        )
        tool_name, tool_input = _parse_tool_call(output)
        if not tool_name:
            return output
        tool_result = await execute_mcp_tool(
            tool_name,
            tool_input,
            brain_pat=brain_pat,
        )
        messages.append({"role": "assistant", "content": output})
        messages.append(
            {
                "role": "user",
                "content": f"Tool '{tool_name}' result:\n{tool_result}",
            }
        )
    return run_inference(
        provider=provider,
        llm_name=llm_name,
        prompt="",
        messages=messages,
        max_tokens=max_tokens,
    )


async def stream_inference_with_mcp_tools(
    *,
    provider: str,
    llm_name: str,
    prompt: str,
    brain_id: str,
    brain_pat: str | None,
    max_tokens: int | None = None,
):
    output = await run_inference_with_mcp_tools(
        provider=provider,
        llm_name=llm_name,
        prompt=prompt,
        brain_id=brain_id,
        brain_pat=brain_pat,
        max_tokens=max_tokens,
    )
    for event in stream_inference_events(
        provider=provider,
        llm_name=llm_name,
        prompt=output,
        max_tokens=max_tokens,
        single_chunk=True,
    ):
        yield event
