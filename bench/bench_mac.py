#!/usr/bin/env python3
"""Time one model on the Mac: seconds per page with Metal and CPU-only, and peak memory.

    python bench/bench_mac.py --run ft-9b-IQ2_M-form --model models/mac/IQ2_M-form.gguf \\
        --mmproj models/mac/mmproj-q8_0.gguf

Starts llama-server (the pinned build in third_party/llama.cpp) under /usr/bin/time -l, sends test
pages one at a time through tinyscribe.client.extract() (same request as the DGX scorer), stops the
server and reads its maximum resident set size (the weights are memory-mapped, so this counts them;
macOS's "peak memory footprint", also kept, leaves them out). Metal: -ngl 99 on
--pages pages; CPU: -ngl 0 on --cpu-pages pages. The first page of each mode is a warm-up and is not
timed. Writes runs/<run>/bench.json, which eval/results_table.py merges into the results table.
Accuracy is not measured here: it comes from the DGX (same weights, same engine).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinyscribe.client import extract, wait_ready  # noqa: E402
from tinyscribe.schema import page_layout  # noqa: E402

SERVER = REPO / "third_party" / "llama.cpp" / "build" / "bin" / "llama-server"
PORT = 8090


def test_pages(manifest: Path, n: int, seed: int = 0) -> list[dict]:
    """n test pages, balanced over scripts, same choice for every model."""
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "test"]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["script"], []).append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        out += [v[i] for v in by.values() if i < len(v)]
        i += 1
    return out[:n]


def run_mode(model: Path, mmproj: Path, ngl: int, pages: list[dict], root: Path, threads: int | None) -> dict:
    cmd = ["/usr/bin/time", "-l", str(SERVER), "-m", str(model), "--mmproj", str(mmproj),
           "-ngl", str(ngl), "--jinja", "-c", "8192", "-np", "1", "--host", "127.0.0.1", "--port", str(PORT)]
    if ngl == 0:
        # CPU-only: no GPU device at all (with -ngl 0 alone llama.cpp still sends big matmuls and the
        # vision encoder to Metal)
        cmd += ["--device", "none", "--no-mmproj-offload", "--no-op-offload"]
    if threads:
        cmd += ["-t", str(threads)]
    import socket
    with socket.socket() as sock:   # a leftover server on the port would answer instead of ours
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} is already in use; stop that server first")
    log_path = REPO / "runs" / "logs" / f"bench_server_{PORT}_ngl{ngl}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    server = f"http://127.0.0.1:{PORT}"
    secs, toks, valid = [], [], 0
    try:
        wait_ready(server)
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early; see {log_path}")
        for k, r in enumerate(pages):
            ex = extract(server, root / r["image"], page_layout(r["template_id"], r["page"]))
            if k == 0:
                continue                            # warm-up
            secs.append(ex.seconds)
            toks.append(ex.completion_tokens or 0)
            valid += ex.data is not None
            print(f"  ngl={ngl} page {k}/{len(pages) - 1}: {ex.seconds:.1f}s, {ex.completion_tokens} tokens",
                  flush=True)
    finally:
        # stop llama-server itself (the child of /usr/bin/time), so time prints its statistics
        kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        for pid in kids:
            subprocess.run(["kill", "-TERM", pid])
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    text = log_path.read_text(errors="replace")

    def stat(name: str) -> int | None:
        m = re.search(rf"(\d+)\s+{name}", text)
        return int(m.group(1)) if m else None
    return {"ngl": ngl, "pages": len(secs), "s_per_page_median": round(statistics.median(secs), 2),
            "s_per_page_mean": round(statistics.mean(secs), 2), "tokens_mean": round(statistics.mean(toks), 1),
            "tokens_per_s": round(sum(toks) / sum(secs), 2), "valid_json": valid / len(secs),
            "peak_rss_bytes": stat("maximum resident set size"),
            "peak_footprint_bytes": stat("peak memory footprint")}


def gb(b: int | None) -> str:
    return "" if b is None else f"{b / 1e9:.2f} GB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name; must match the DGX run for the results table")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.jsonl"))
    ap.add_argument("--pages", type=int, default=21, help="Metal pages, first one is warm-up")
    ap.add_argument("--cpu-pages", type=int, default=6, help="CPU-only pages, first one is warm-up (0 = skip)")
    ap.add_argument("--threads", type=int, help="CPU threads (default: llama.cpp's choice)")
    a = ap.parse_args(argv)
    manifest = Path(a.manifest)
    (REPO / "runs" / "logs").mkdir(parents=True, exist_ok=True)
    out = {"model": a.model, "mmproj": a.mmproj, "model_bytes": Path(a.model).stat().st_size,
           "mmproj_bytes": Path(a.mmproj).stat().st_size, "date": time.strftime("%Y-%m-%d"),
           "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip(),
           "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                              text=True).stdout.strip() or 0)}
    metal = run_mode(Path(a.model), Path(a.mmproj), 99, test_pages(manifest, a.pages), manifest.parent, a.threads)
    out["metal"] = metal
    out["s_per_page_metal"] = f"{metal['s_per_page_median']:.1f}"
    out["peak_memory"] = gb(metal["peak_rss_bytes"])
    if a.cpu_pages:
        cpu = run_mode(Path(a.model), Path(a.mmproj), 0, test_pages(manifest, a.cpu_pages), manifest.parent,
                       a.threads)
        out["cpu"] = cpu
        out["s_per_page_cpu"] = f"{cpu['s_per_page_median']:.1f}"
    run_dir = REPO / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "bench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("s_per_page_metal", "s_per_page_cpu", "peak_memory") if k in out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
