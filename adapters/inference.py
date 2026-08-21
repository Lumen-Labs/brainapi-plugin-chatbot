import json
import os
import time
import uuid
from collections.abc import Generator

from src.config import _PROVIDERS, config


PROVIDER_ALIASES = {
    "azureopenai": "azure",
    "azure_openai": "azure",
    "claude": "anthropic",
    "bedrock": "amazon_bedrock",
    "vertex": "gcp_vertex",
    "gcp": "gcp_vertex",
}


def supported_providers() -> set[str]:
    return set(_PROVIDERS) | set(PROVIDER_ALIASES.keys())


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[normalized]
    return normalized


def parse_model(model: str) -> tuple[str, str]:
    provider_raw, sep, llm_name = model.strip().partition("::")
    if sep != "::" or not provider_raw or not llm_name:
        raise ValueError("model must be in 'api_provider::llm_name' format")
    provider = normalize_provider(provider_raw)
    if provider not in set(_PROVIDERS):
        raise ValueError(f"unsupported provider '{provider_raw}'")
    return provider, llm_name.strip()


def _build_openai_compatible_client(provider: str):
    from openai import OpenAI, AzureOpenAI

    if provider == "openai":
        api_key = (
            getattr(config.openai, "api_key", None)
            if config.openai is not None
            else os.getenv("OPENAI_API_KEY")
        )
        base_url = (
            getattr(config.openai, "base_url", None)
            if config.openai is not None
            else os.getenv("OPENAI_BASE_URL")
        )
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for provider 'openai' in chatbot inference"
            )
        kwargs = {
            "api_key": api_key,
            "timeout": 120.0,
            "max_retries": 3,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)
    if provider == "azure":
        return AzureOpenAI(
            api_version=config.azure.llm_api_version,
            azure_endpoint=config.azure.llm_endpoint,
            api_key=config.azure.llm_subscription_key,
        )
    if provider == "ollama":
        return OpenAI(
            base_url=f"http://{config.ollama.host}:{config.ollama.port}/v1/",
            api_key="ollama",
            timeout=120.0,
            max_retries=1,
        )
    return None


def _build_anthropic_client(provider: str):
    if provider != "anthropic":
        return None
    import importlib

    anthropic_module = importlib.import_module("anthropic")
    api_key = (
        getattr(config.anthropic, "api_key", None)
        if config.anthropic is not None
        else os.getenv("ANTHROPIC_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required for provider 'anthropic' in chatbot inference"
        )
    return anthropic_module.Anthropic(api_key=api_key)


def run_inference(
    provider: str,
    llm_name: str,
    prompt: str,
    max_tokens: int | None = None,
    messages: list[dict[str, str]] | None = None,
) -> str:
    from src.core.instances import _build_small_llm

    chat_messages = messages if messages is not None else [{"role": "user", "content": prompt}]
    client = _build_openai_compatible_client(provider)
    if client is not None:
        response = client.chat.completions.create(
            model=llm_name,
            messages=chat_messages,
            **({"max_tokens": max_tokens} if max_tokens else {}),
        )
        return response.choices[0].message.content or ""
    anthropic_client = _build_anthropic_client(provider)
    if anthropic_client is not None:
        response = anthropic_client.messages.create(
            model=llm_name,
            max_tokens=max_tokens or 1024,
            messages=chat_messages,
        )
        parts = [block.text for block in response.content if getattr(block, "text", None)]
        return "".join(parts)

    llm_client = _build_small_llm(provider)
    original_model = getattr(llm_client, "model", None)
    try:
        setattr(llm_client, "model", llm_name)
        return llm_client.generate_text(prompt, max_tokens)
    finally:
        if original_model is not None:
            setattr(llm_client, "model", original_model)


def stream_inference_events(
    provider: str,
    llm_name: str,
    prompt: str,
    max_tokens: int | None = None,
    single_chunk: bool = False,
) -> Generator[str, None, None]:
    if single_chunk:
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        payload = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": f"{provider}::{llm_name}",
            "choices": [
                {"index": 0, "delta": {"content": prompt}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        return
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    client = _build_openai_compatible_client(provider)
    if client is not None:
        stream = client.chat.completions.create(
            model=llm_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **({"max_tokens": max_tokens} if max_tokens else {}),
        )
        for chunk in stream:
            delta = ""
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
            payload = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": f"{provider}::{llm_name}",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": (
                            chunk.choices[0].finish_reason if chunk.choices else None
                        ),
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        return
    anthropic_client = _build_anthropic_client(provider)
    if anthropic_client is not None:
        with anthropic_client.messages.stream(
            model=llm_name,
            max_tokens=max_tokens or 1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                payload = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": f"{provider}::{llm_name}",
                    "choices": [
                        {"index": 0, "delta": {"content": text}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    output = run_inference(provider, llm_name, prompt, max_tokens=max_tokens)
    payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": f"{provider}::{llm_name}",
        "choices": [{"index": 0, "delta": {"content": output}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"
