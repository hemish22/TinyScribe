"""Client for llama.cpp's llama-server. The DGX scorer, the Mac benchmark, the CLI and the web demo
all call extract(), so every number comes from the same request: same image bytes, prompt,
grammar (from the page's JSON Schema) and sampling.

The server must run with --jinja and --mmproj; thinking is switched off per request."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from tinyscribe.prompt import build_prompt
from tinyscribe.schema import PageLayout, json_schema

MAX_TOKENS = 4096
TIMEOUT_S = 600


@dataclass
class Extraction:
    data: dict | None        # parsed JSON, None if the output did not parse
    raw: str                 # model text exactly as returned
    seconds: float           # wall time of the request
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def request_body(image: bytes, lay: PageLayout, max_tokens: int = MAX_TOKENS) -> dict:
    mime = "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"
    url = f"data:{mime};base64," + base64.b64encode(image).decode()
    return {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": build_prompt(lay)},
        ]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": lay.schema_name, "schema": json_schema(lay)}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract(server: str, image: Path | bytes, lay: PageLayout, max_tokens: int = MAX_TOKENS) -> Extraction:
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    t0 = time.perf_counter()
    r = requests.post(f"{server.rstrip('/')}/v1/chat/completions",
                      json=request_body(data, lay, max_tokens), timeout=TIMEOUT_S)
    secs = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    raw = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage", {})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    return Extraction(data, raw, secs, usage.get("prompt_tokens"), usage.get("completion_tokens"))


def wait_ready(server: str, timeout_s: float = 900) -> None:
    """Block until llama-server has loaded the model (GET /health returns 200)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(f"{server.rstrip('/')}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"{server} not ready after {timeout_s:.0f}s")
