from collections.abc import AsyncGenerator, Generator

from adapters.inference import parse_model, run_inference, stream_inference_events
from adapters.mcp_inference import (
    run_inference_with_mcp_tools,
    stream_inference_with_mcp_tools,
)


async def inference_response(
    model: str,
    prompt: str,
    stream: bool = False,
    max_tokens: int | None = None,
    *,
    brain_id: str | None = None,
    brain_pat: str | None = None,
    use_mcp_tools: bool = True,
) -> tuple[str, str, str | Generator[str, None, None] | AsyncGenerator[str, None]]:
    provider, llm_name = parse_model(model)
    use_mcp = use_mcp_tools and brain_id
    if stream:
        if use_mcp:
            return (
                provider,
                llm_name,
                stream_inference_with_mcp_tools(
                    provider=provider,
                    llm_name=llm_name,
                    prompt=prompt,
                    brain_id=brain_id,
                    brain_pat=brain_pat,
                    max_tokens=max_tokens,
                ),
            )
        return (
            provider,
            llm_name,
            stream_inference_events(
                provider=provider,
                llm_name=llm_name,
                prompt=prompt,
                max_tokens=max_tokens,
            ),
        )
    if use_mcp:
        output = await run_inference_with_mcp_tools(
            provider=provider,
            llm_name=llm_name,
            prompt=prompt,
            brain_id=brain_id,
            brain_pat=brain_pat,
            max_tokens=max_tokens,
        )
        return provider, llm_name, output
    output = run_inference(
        provider=provider,
        llm_name=llm_name,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return provider, llm_name, output
