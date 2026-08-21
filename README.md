# chatbot

Official BrainAPI inference plugin. It adds `POST /chatbot/inference` so a deployed BrainAPI instance can talk to the LLM providers you already configured — with optional streaming, MCP tool calling, and conversation memory when [chatbot-memory](https://github.com/Lumen-Labs/brainapi-plugin-chatbot-memory) is installed.

| | |
|---|---|
| Registry name | `chatbot` |
| Version | `1.0.0` |
| BrainAPI | `>=2.13.0` |
| Route prefix | `/chatbot` |
| Extra pip deps | none |

Product docs: [Chatbot](https://brainapi.lumen-labs.ai/docs/v2/chatbot).

## What it does

- Resolves `model` as `api_provider::llm_name` against BrainAPI’s runtime provider registry.
- Runs a single-turn or streamed completion against OpenAI-compatible, Azure, Ollama, Anthropic, or core small-LLM adapters (DeepSeek, Vertex, Bedrock, …).
- When a `brain_id` is present, prepends MCP tool instructions and may iterate tool calls (default max 5).
- If `plugins/chatbot-memory` is loaded **and** the request includes `conversation_id`, it loads prior turns / summary / preferences, saves the user message, generates, then saves the agent reply.

It does not replace `/retrieve/context`. Memory is opt-in via the sibling plugin.

## Install

```bash
git clone https://github.com/Lumen-Labs/brainapi-plugin-chatbot.git plugins/chatbot
```

Or:

```bash
./bin/brainapi install chatbot
```

Restart the API. Restart the MCP server as well if you expect tool calling.

To persist conversations, also install [chatbot-memory](https://github.com/Lumen-Labs/brainapi-plugin-chatbot-memory) into `plugins/chatbot-memory` (that directory name is what this plugin looks for).

## Quick start

Auth is the same as other BrainAPI routes (`BrainPAT` or `Authorization: Bearer …`, plus brain scoping / `X-Brain-ID`).

```bash
curl -X POST "$BRAINAPI_URL/chatbot/inference" \
  -H "Content-Type: application/json" \
  -H "BrainPAT: $BRAINPAT_TOKEN" \
  -H "X-Brain-ID: example01" \
  -d '{
    "model": "openai::gpt-4o-mini",
    "input": "Write a one-line greeting",
    "stream": false,
    "max_tokens": 64
  }'
```

Non-stream response:

```json
{
  "message": "Inference completed successfully",
  "data": {
    "model": "openai::gpt-4o-mini",
    "provider": "openai",
    "output": "Hello there!",
    "stream": false
  }
}
```

Invalid `model` format or unsupported provider → **422**.

## Request body

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | `api_provider::llm_name` |
| `input` | string | yes | User message |
| `stream` | bool | no | `true` → `text/event-stream` (OpenAI-style SSE) |
| `max_tokens` | int | no | Generation cap passed to the provider |
| `conversation_id` | string | no | Load/save memory when chatbot-memory is installed |
| `user_id` | string | no | Preference lookup via chatbot-memory |

### Model string

`model` must contain `::`. Provider aliases:

| Alias | Resolves to |
|---|---|
| `azureopenai`, `azure_openai` | `azure` |
| `claude` | `anthropic` |
| `bedrock` | `amazon_bedrock` |
| `vertex`, `gcp` | `gcp_vertex` |

Configured providers typically include `ollama`, `azure`, `openai`, `anthropic`, `deepseek`, `gcp_vertex`, `amazon_bedrock`. Keys and endpoints come from the same `.env` as core LLM adapters.

## Streaming

Set `"stream": true`.

- Content-Type: `text/event-stream`
- Chunks: `data: {...}` lines, ended by `data: [DONE]`
- Header `X-Model: provider::llm_name`
- Cache-Control: `no-cache`

When MCP tools are used, the tool loop runs **first**; the stream then emits the final answer as a single SSE chunk (`single_chunk=true`).

## Memory integration

Detection: this plugin looks for a sibling directory named **`chatbot-memory`** (not the registry name `chatbot-memory-single-brain`). Clone or install into that folder.

When memory is available and `conversation_id` is set:

1. `get_chatbot_memory_conversation_context` loads last messages, conversation meta summary, and user preferences (`user_id`).
2. The user turn is saved **before** generation.
3. The agent turn is saved after generation (including streamed replies, once the stream finishes).

Without chatbot-memory, `conversation_id` does not load a structured memory pack. Bare text chunks are not written through the memory pipeline.

## MCP tool calling

Enabled when `brain_id` is available (default). The agent uses the same BrainPAT as the HTTP request to execute tools.

| Env | Default | Meaning |
|---|---|---|
| `CHATBOT_MCP_TOOL_MAX_ITERATIONS` | `5` | Max tool-call round trips per request |

The MCP **server** must be running. This is BrainAPI’s product MCP, not the docs MCP at `/docs/mcp`.

## Layout

```text
chatbot/
  plugin.yaml
  main.py                 # memory detection + include_router
  routes/inference.py     # POST /chatbot/inference
  controllers/inference.py
  adapters/inference.py   # provider clients + SSE
  adapters/mcp_inference.py
```

## Publishing

Pushes to `main` publish this package to the BrainAPI registry via GitHub Actions. Manual run: Actions → **Publish to BrainAPI registry**.

## License

Business Source License 1.1. See [LICENSE](LICENSE).

## Related

- [chatbot-memory](https://github.com/Lumen-Labs/brainapi-plugin-chatbot-memory)
- [BrainAPI](https://github.com/Lumen-Labs/brainapi2)
- [Plugins](https://brainapi.lumen-labs.ai/docs/plugins)
- [MCP](https://brainapi.lumen-labs.ai/docs/v2/agentic/MCP)
