"""OpenAI-compatible chat client for the local vLLM server.

Only the pieces the pipeline needs: text + one image per request, a forced
tool call (query generation) and JSON-schema constrained output (verdicts).
Every call returns the parsed payload together with the raw message, token
usage and latency so a run file shows exactly what the model produced.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import httpx

from .config import Config

_MAGIC = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]


class VLMError(RuntimeError):
    """The model server failed or returned something the pipeline cannot use."""


def image_mime(data: bytes) -> str:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported or unreadable image (expected JPEG, PNG, GIF, BMP or WebP)")


def image_data_url(path: str | Path, max_bytes: int) -> str:
    data = Path(path).read_bytes()
    if len(data) > max_bytes:
        raise ValueError(f"image is {len(data)} bytes, above the {max_bytes}-byte limit")
    return f"data:{image_mime(data)};base64,{base64.b64encode(data).decode()}"


def user_content(text: str, image_url: str | None) -> list[dict]:
    content: list[dict] = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": text})
    return content


def extract_json(text: str):
    """Parse a JSON object from model text, tolerating code fences or a preamble."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object in model output", text, 0)


class VLMClient:
    def __init__(self, cfg: Config, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=cfg.vlm_timeout)
        self.base = cfg.vlm_url.rstrip("/")

    def models(self) -> list[dict]:
        r = self.http.get(f"{self.base}/models", timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def chat(self, messages: list[dict], *, tools=None, tool_choice=None,
             response_format=None, max_tokens: int | None = None) -> dict:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": max_tokens or self.cfg.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if response_format:
            body["response_format"] = response_format
        t0 = time.perf_counter()
        try:
            r = self.http.post(f"{self.base}/chat/completions", json=body, timeout=self.cfg.vlm_timeout)
        except httpx.HTTPError as e:
            raise VLMError(f"model server unreachable: {type(e).__name__}: {e}") from e
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            raise VLMError(f"model server HTTP {r.status_code}: {r.text[:500]}")
        payload = r.json()
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError) as e:
            raise VLMError(f"malformed completion: {str(payload)[:500]}") from e
        return {
            "message": choice.get("message", {}),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage", {}),
            "latency_s": round(latency, 3),
        }

    def tool_arguments(self, messages: list[dict], tool: dict) -> tuple[dict, dict]:
        """Force one call of ``tool``; return (arguments, raw call record)."""
        name = tool["function"]["name"]
        out = self.chat(messages, tools=[tool],
                        tool_choice={"type": "function", "function": {"name": name}})
        msg = out["message"]
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") == name:
                args = fn.get("arguments") or "{}"
                try:
                    return (json.loads(args) if isinstance(args, str) else args), out
                except json.JSONDecodeError as e:
                    raise VLMError(f"tool arguments are not JSON: {args[:300]}") from e
        # Some parsers leave the call in the content; accept it only if it parses.
        try:
            return extract_json(msg.get("content") or ""), out
        except json.JSONDecodeError:
            raise VLMError(f"model did not call {name}: {str(msg.get('content'))[:300]}") from None

    def json_object(self, messages: list[dict], name: str, schema: dict) -> tuple[dict, dict]:
        """Request JSON constrained by ``schema``; return (object, raw call record)."""
        out = self.chat(messages, response_format={
            "type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}})
        if out["finish_reason"] == "length":
            raise VLMError("model output was cut off at max_tokens")
        try:
            return extract_json(out["message"].get("content") or ""), out
        except json.JSONDecodeError:
            raise VLMError(f"model output is not JSON: {str(out['message'].get('content'))[:300]}") from None
