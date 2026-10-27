#!/usr/bin/env python3
"""Score one model on the test split through llama-server(s).

    python eval/run_eval.py --name zero-shot-9b --label "9B zero-shot, BF16 (DGX)" \
        --servers http://127.0.0.1:8081,http://127.0.0.1:8082 --slots 2

Pages are spread over every server x slot. Predictions stream to runs/<name>/preds.jsonl
(resumable: pages already there are skipped), then runs/<name>/scores.json is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinyscribe.client import extract, wait_ready
from tinyscribe.prompt import PROMPT_VERSION
from tinyscribe.schema import json_schema, page_layout
from tinyscribe.score import score_records


def load_rows(manifest: Path, split: str, limit: int | None) -> list[dict]:
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == split]
    if limit:  # keep the per-script balance when sampling
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["script"], []).append(r)
        rows = [r for rs in by.values() for r in rs[: max(1, limit // len(by))]]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run directory under runs/")
    ap.add_argument("--label", help="row label for the results table")
    ap.add_argument("--servers", default="http://127.0.0.1:8081")
    ap.add_argument("--slots", type=int, default=1, help="parallel requests per server (llama-server -np)")
    ap.add_argument("--manifest", default=str(REPO / "data" / "manifest.jsonl"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, help="score only this many pages (balanced per script)")
    ap.add_argument("--meta", default="{}", help="extra JSON stored in meta.json (size, quant, imatrix...)")
    a = ap.parse_args(argv)

    manifest = Path(a.manifest)
    data_root = manifest.parent
    run = REPO / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    servers = [s.strip() for s in a.servers.split(",") if s.strip()]
    for s in servers:
        wait_ready(s)

    rows = load_rows(manifest, a.split, a.limit)
    preds_path = run / "preds.jsonl"
    # a page counts as done only with a model answer; failed requests (server down, timeout) are retried.
    # The last line for a page wins below, so a retry replaces the failed record.
    done: set[str] = set()
    if preds_path.exists():
        for x in preds_path.read_text().splitlines():
            p = json.loads(x)
            (done.discard if "error" in p else done.add)(p["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.name}: {len(rows)} pages, {len(done)} done, {len(todo)} to run on {len(servers)} server(s)")

    lock = threading.Lock()
    t0 = time.time()
    n = 0

    def work(i_row):
        nonlocal n
        i, r = i_row
        server = servers[i % len(servers)]
        lay = page_layout(r["template_id"], r["page"])
        try:
            ex = extract(server, data_root / r["image"], lay)
            rec = {"id": r["id"], "pred": ex.data, "raw": ex.raw, "seconds": round(ex.seconds, 3),
                   "prompt_tokens": ex.prompt_tokens, "completion_tokens": ex.completion_tokens}
        except Exception as e:  # recorded and retried on the next run; the run exits non-zero meanwhile
            rec = {"id": r["id"], "pred": None, "raw": "", "error": repr(e)}
        with lock:
            with preds_path.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)}  {rate:.2f} pages/s", flush=True)

    with ThreadPoolExecutor(len(servers) * a.slots) as ex:
        list(ex.map(work, enumerate(todo)))

    preds = {}
    for x in preds_path.read_text().splitlines():
        p = json.loads(x)
        preds[p["id"]] = p
    records = [{"script": r["script"], "target": r["target"], "pred": preds[r["id"]]["pred"],
                "schema": json_schema(page_layout(r["template_id"], r["page"]))}
               for r in rows if r["id"] in preds]
    report = score_records(records).as_dict()
    errors = sum(1 for r in rows if "error" in preds.get(r["id"], {}))
    report["errors"] = errors
    (run / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    meta = {"name": a.name, "label": a.label or a.name, "split": a.split, "pages": len(records),
            "prompt_version": PROMPT_VERSION, "manifest": str(manifest), **json.loads(a.meta)}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(report["overall"]), f"errors={errors}")
    if errors:
        # a failed request is not a model answer: these scores count it as an empty page. Re-run the
        # same command to retry only the failed pages.
        print(f"ERROR: {errors} page(s) failed to get an answer; scores are not final. Re-run to retry them.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
