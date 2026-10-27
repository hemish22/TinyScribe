"""Start and stop a local llama-server for the CLI and the web demo.

    with LlamaServer("9b") as srv:
        extract(srv.url, image, layout)

Model presets point at the files the Mac benchmark timed (models/mac/); a path works too. The
server runs with the benchmark's settings, so speed and memory match the README table.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

from tinyscribe.client import wait_ready

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models" / "mac"

# name -> (model, vision projector, label)
PRESETS = {
    "9b": (MODELS / "ft-9b" / "IQ3_XXS_oq4_K-form.gguf", MODELS / "ft-9b" / "mmproj-q8_0.gguf",
           "TinyScribe 9B, IQ3_XXS (3.9 GB)"),
    "2b": (MODELS / "ft-2b" / "Q8_0-none.gguf", MODELS / "ft-2b" / "mmproj-q8_0.gguf",
           "TinyScribe 2B, Q8_0 (2.0 GB)"),
}
DEFAULT = os.environ.get("TINYSCRIBE_MODEL", "9b")


def resolve(model: str, mmproj: str | None = None) -> tuple[Path, Path, str]:
    """Preset name or GGUF path -> (model, projector, label). A path needs --mmproj, or a
    mmproj*.gguf next to it."""
    if model in PRESETS:
        m, p, label = PRESETS[model]
        p = Path(mmproj) if mmproj else p
    else:
        m = Path(model)
        cands = sorted(m.parent.glob("mmproj*.gguf"))
        if mmproj:
            p = Path(mmproj)
        elif cands:
            p = cands[0]
        else:
            raise FileNotFoundError(f"no mmproj*.gguf next to {m}; pass --mmproj")
        label = m.name
    for f in (m, p):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found (presets expect the files in {MODELS})")
    return m, p, label


def server_binary() -> str:
    env = os.environ.get("LLAMA_SERVER")
    built = REPO / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
    for c in (env, str(built) if built.exists() else None, shutil.which("llama-server")):
        if c:
            return c
    raise FileNotFoundError("llama-server not found: run `make llama`, or set LLAMA_SERVER")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, model: str = DEFAULT, mmproj: str | None = None, port: int | None = None,
                 cpu: bool = False, log: Path | None = None):
        self.model, self.mmproj, self.label = resolve(model, mmproj)
        self.port = port or free_port()
        self.cpu = cpu
        self.log_path = log or REPO / "runs" / "logs" / f"tinyscribe_server_{self.port}.log"
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cmd = [server_binary(), "-m", str(self.model), "--mmproj", str(self.mmproj), "--jinja",
               "-c", "8192", "-np", "1", "--host", "127.0.0.1", "--port", str(self.port),
               "--cache-ram", "0"]   # a prompt cache never helps: every page's image tokens differ
        if self.cpu:
            cmd += ["-ngl", "0", "--device", "none", "--no-mmproj-offload", "--no-op-offload"]
        else:
            cmd += ["-ngl", "99"]
        return cmd

    def start(self) -> "LlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT)
        for _ in range(60):   # up to ~5 min to load
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {self.log_path}")
            try:
                wait_ready(self.url, timeout_s=5)
                return self
            except TimeoutError:
                pass
        self.stop()
        raise TimeoutError(f"llama-server not ready after 5 min; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
