import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from controllers.inference import inference_response
from main import memory_plugin_available
from src.constants.data import BRAIN_VERSION, TextChunk
from src.core.plugins.context import PluginContext
from src.services.api.dependencies import get_brain_id


class InferenceRequest(BaseModel):
    model: str
    input: str
    stream: bool = False
    max_tokens: int | None = None
    conversation_id: str | None = None
    user_id: str | None = None


async def _save_chatbot_message(
    context: PluginContext,
    brain_id: str,
    conversation_id: str,
    role: str,
    message: str,
    user_id: str | None = None,
):
    text_chunk = TextChunk(
        id=f"chatbot-{uuid.uuid4()}",
        text=f"{role}: {message}",
        metadata={
            "role": role.lower(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "kind": "chatbot_message",
        },
        brain_version=BRAIN_VERSION,
    )
    if not memory_plugin_available:
        await asyncio.to_thread(
            context.adapters.data.save_text_chunk, text_chunk, brain_id=brain_id
        )
        return
    save_with_memory_controller = getattr(context, "save_chatbot_memory_message", None)
    if callable(save_with_memory_controller):
        await save_with_memory_controller(context, text_chunk, brain_id)
        return
    await asyncio.to_thread(
        context.adapters.data.save_text_chunk, text_chunk, brain_id=brain_id
    )


async def _build_prompt_with_context(
    context: PluginContext,
    brain_id: str,
    prompt: str,
    conversation_id: str | None,
    user_id: str | None = None,
):
    if not memory_plugin_available or not conversation_id:
        return prompt
    get_conversation_context = getattr(
        context, "get_chatbot_memory_conversation_context", None
    )
    full_prompt = prompt
    if callable(get_conversation_context):
        conversation_context = await get_conversation_context(
            ctx=context,
            conversation_id=conversation_id,
            brain_id=brain_id,
            user_id=user_id,
        )
        chunks = conversation_context.get("last_messages") or []
        meta = conversation_context.get("meta")
        preferences = conversation_context.get("preferences")
        last_messages_text = "\n".join(chunk.text for chunk in chunks)
        meta_text = meta.summary if meta else ""
        preferences_text = preferences.preferences if preferences else ""
        full_prompt = f"Previous messages: <start_of_previous_messages>{last_messages_text}<end_of_previous_messages>\n\nCurrent new user message: <start_of_new_user_message>{prompt}<end_of_new_user_message>\n\nConversation meta: <start_of_conversation_meta>{meta_text}<end_of_conversation_meta>\n\nPreferences: <start_of_preferences>{preferences_text}<end_of_preferences>"
    else:
        chunks, _ = await asyncio.to_thread(
            context.adapters.data.get_text_chunks,
            brain_id,
            20,
            0,
            None,
            {"conversation_id": conversation_id},
            "asc",
        )
        context_lines = "\n".join(chunk.text for chunk in chunks)
        full_prompt = f"Previous messages: <start_of_previous_messages>{context_lines}<end_of_previous_messages>\n\nCurrent new user message: <start_of_new_user_message>{prompt}<end_of_new_user_message>"
    if not chunks:
        return prompt
    return full_prompt


def _get_brainpat(request: Request) -> str | None:
    brainpat = request.headers.get("BrainPAT")
    if brainpat:
        return brainpat.rstrip()
    authorization = request.headers.get("Authorization")
    if authorization and " " in authorization:
        return authorization.split(" ", 1)[1].strip() or None
    return None


def create_inference_router(context: PluginContext) -> APIRouter:
    router = APIRouter(prefix="/chatbot", tags=["chatbot"])

    @router.post(path="/inference")
    async def infer(
        request: InferenceRequest,
        http_request: Request,
        brain_id: str = Depends(get_brain_id),
    ):
        try:
            prompt = await _build_prompt_with_context(
                context=context,
                brain_id=brain_id,
                prompt=request.input,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
            )
            if memory_plugin_available and request.conversation_id:
                await _save_chatbot_message(
                    context=context,
                    brain_id=brain_id,
                    conversation_id=request.conversation_id,
                    role="User",
                    message=request.input,
                    user_id=request.user_id,
                )
            brain_pat = _get_brainpat(http_request)
            if request.stream:
                provider, llm_name, event_stream = await inference_response(
                    model=request.model,
                    prompt=prompt,
                    stream=True,
                    max_tokens=request.max_tokens,
                    brain_id=brain_id,
                    brain_pat=brain_pat,
                )

                async def wrapped_stream():
                    output_chunks: list[str] = []
                    try:
                        async for event in event_stream:
                            if event.startswith("data: "):
                                payload = event.removeprefix("data: ").strip()
                                if payload and payload != "[DONE]":
                                    try:
                                        parsed = json.loads(payload)
                                        choices = parsed.get("choices") or []
                                        if choices:
                                            delta = choices[0].get("delta") or {}
                                            content = delta.get("content")
                                            if content:
                                                output_chunks.append(content)
                                    except (
                                        json.JSONDecodeError,
                                        ValueError,
                                        TypeError,
                                    ):
                                        pass
                            yield event
                    finally:
                        if memory_plugin_available and request.conversation_id:
                            agent_output = "".join(output_chunks).strip()
                            if agent_output:
                                await _save_chatbot_message(
                                    context=context,
                                    brain_id=brain_id,
                                    conversation_id=request.conversation_id,
                                    role="Agent",
                                    message=agent_output,
                                    user_id=request.user_id,
                                )

                return StreamingResponse(
                    wrapped_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Model": f"{provider}::{llm_name}",
                    },
                )

            provider, llm_name, output = await inference_response(
                model=request.model,
                prompt=prompt,
                stream=False,
                max_tokens=request.max_tokens,
                brain_id=brain_id,
                brain_pat=brain_pat,
            )
            if memory_plugin_available and request.conversation_id:
                await _save_chatbot_message(
                    context=context,
                    brain_id=brain_id,
                    conversation_id=request.conversation_id,
                    role="Agent",
                    message=output,
                    user_id=request.user_id,
                )
            return {
                "message": "Inference completed successfully",
                "data": {
                    "model": f"{provider}::{llm_name}",
                    "provider": provider,
                    "output": output,
                    "stream": False,
                },
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router
