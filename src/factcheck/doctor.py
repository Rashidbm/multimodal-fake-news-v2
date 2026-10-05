"""Backend readiness check: model server, image input, tool calling, JSON output, search.

This talks to the real services.  It proves they answer and parse; it says
nothing about accuracy.
"""
from __future__ import annotations

import base64
import struct
import zlib

from .config import Config
from .pipeline import search_tool, verdict_schema
from .retrieval import SearchError, SearxClient
from .vlm import VLMClient, user_content


def tiny_png(size: int = 32) -> str:
    """A solid red PNG as a data URL, built without Pillow."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def run_doctor(cfg: Config, vlm: VLMClient | None = None, searcher: SearxClient | None = None, log=print) -> bool:
    vlm = vlm or VLMClient(cfg)
    searcher = searcher or SearxClient(cfg)
    checks: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append((name, True, detail))
        except Exception as e:  # noqa: BLE001 - report every failure
            checks.append((name, False, f"{type(e).__name__}: {e}"[:400]))
        ok, detail = checks[-1][1], checks[-1][2]
        log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        return ok

    def models():
        listed = vlm.models()
        ids = [m.get("id") for m in listed]
        if cfg.model not in ids:
            raise RuntimeError(f"{cfg.model!r} not served; server lists {ids}")
        m = next(m for m in listed if m.get("id") == cfg.model)
        return f"{cfg.model} served, max_model_len={m.get('max_model_len')}, root={m.get('root')}"

    def text_chat():
        out = vlm.chat([{"role": "user", "content": "Reply with the single word READY."}], max_tokens=16)
        return f"{(out['message'].get('content') or '').strip()[:40]!r} in {out['latency_s']}s"

    def image_chat():
        msgs = [{"role": "user", "content": user_content(
            "What is the main colour of this image? Answer with one word.", tiny_png())}]
        out = vlm.chat(msgs, max_tokens=16)
        answer = (out["message"].get("content") or "").strip()
        if "red" not in answer.lower():
            raise RuntimeError(f"expected 'red', got {answer[:60]!r}")
        return f"{answer[:40]!r} in {out['latency_s']}s"

    def tool_call():
        args, raw = vlm.tool_arguments(
            [{"role": "user", "content": "Find recent reporting on the Eiffel Tower's height."}], search_tool(3))
        queries = args.get("queries")
        if not isinstance(queries, list) or not queries:
            raise RuntimeError(f"tool arguments without queries: {args}")
        return f"{queries} in {raw['latency_s']}s"

    def json_output():
        obj, raw = vlm.json_object([{"role": "system", "content": "Answer with JSON only."},
                                    {"role": "user", "content": "Claim: water boils at 100 C at sea level. "
                                     "Label it with an empty citations list."}], "verdict", verdict_schema("claim"))
        return f"label={obj.get('label')} in {raw['latency_s']}s"

    def search():
        res = searcher.search("Eiffel Tower height")
        if not res["results"]:
            raise SearchError(f"no results; unresponsive engines: {res['unresponsive_engines']}")
        note = f", unresponsive: {res['unresponsive_engines']}" if res["unresponsive_engines"] else ""
        return f"{len(res['results'])} results, first {res['results'][0]['url']}{note}"

    if check("model server", models):
        check("text chat", text_chat)
        check("image input", image_chat)
        check("forced tool call", tool_call)
        check("JSON-schema output", json_output)
    check("SearXNG JSON search", search)

    ok = all(c[1] for c in checks)
    log("\nAll checks passed. This confirms the services respond, not that fact checks are correct."
        if ok else "\nSome checks failed; fix them before running experiments.")
    return ok

