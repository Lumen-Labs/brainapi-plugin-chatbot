## Chatbot Inference Endpoint

`chatbot` exposes a single inference endpoint:

- `POST /chatbot/inference`

Request body:

```json
{
  "model": "azureopenai::gpt-4o-mini",
  "input": "Write a one-line greeting",
  "stream": false,
  "max_tokens": 64
}
```

`model` must be in `api_provider::llm_name` format. Supported providers are resolved from the runtime provider registry and include aliases such as `azureopenai` for `azure`.

Non-stream response:

```json
{
  "message": "Inference completed successfully",
  "data": {
    "model": "azure::gpt-4o-mini",
    "provider": "azure",
    "output": "Hello there!",
    "stream": false
  }
}
```

Stream response:

- Set `"stream": true`
- Response content type is `text/event-stream`
- Events are emitted in OpenAI-style `data: {...}` chunks and end with `data: [DONE]`
