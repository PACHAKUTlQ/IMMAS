# Backend interface

Module: `immas.router.backend`

## Contract

The router expects an OpenAI-compatible backend that supports:

- `GET /v1/models`
- `POST /v1/chat/completions`

The backend forwarder:

- sends JSON body unchanged
- forwards selected headers (including authorization)
- does not support streaming passthrough (currently)

Timeout:

- `timeout=None` for chat completions forward call (wait indefinitely)
