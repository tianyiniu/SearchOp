"""vLLM middleware: give gpt-oss chat requests a default `reasoning_effort`.

vLLM builds gpt-oss prompts through its harmony path, which reads
`reasoning_effort` only from the request body; there is no server flag for a
default, and --default-chat-template-kwargs is ignored for this model. This
fills the field in when the client left it out, so a request that sets its own
value still wins.

Loaded by deploy_gpt_oss_20b.sh via
  --middleware gpt_oss_default_effort.DefaultReasoningEffort
The default comes from GPT_OSS_REASONING_EFFORT (low | medium | high).
"""

import json
import os

DEFAULT_EFFORT = os.environ.get("GPT_OSS_REASONING_EFFORT", "low")
PATHS = ("/v1/chat/completions",)


class DefaultReasoningEffort:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in PATHS
        ):
            return await self.app(scope, receive, send)

        # Read the whole request body before the app sees it.
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # Client went away; let the app handle the disconnect.
                first = message
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                first = None
                break

        if first is None:
            try:
                data = json.loads(body)
                if isinstance(data, dict) and data.get("reasoning_effort") is None:
                    data["reasoning_effort"] = DEFAULT_EFFORT
                    body = json.dumps(data).encode()
            except ValueError:
                pass  # not JSON: pass it through and let vLLM report the error

            headers = [
                (k, v) for k, v in scope["headers"] if k.lower() != b"content-length"
            ]
            headers.append((b"content-length", str(len(body)).encode()))
            scope = dict(scope, headers=headers)
            first = {"type": "http.request", "body": body, "more_body": False}

        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return first
            return await receive()

        await self.app(scope, replay, send)
