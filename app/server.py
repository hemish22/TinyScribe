#!/usr/bin/env python3
"""Local web demo: drop a form page, pick its form, read it into JSON. Everything stays on this machine.

    python app/server.py                 # 9B IQ3_XXS, http://127.0.0.1:8000
    python app/server.py --model 2b      # faster, 2B Q8_0
    make demo [MODEL=2b]

Starts llama-server (tinyscribe.server) and a small stdlib HTTP server for app/index.html. Test pages
from data/ are offered as samples; for those the page's ground truth is shown next to each value.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import signal
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tinyscribe.client import extract  # noqa: E402
from tinyscribe.image import prepare  # noqa: E402
from tinyscribe.normalize import normalize  # noqa: E402
from tinyscribe.schema import load_template, page_layout, page_layouts, parse_schema_name, template_ids  # noqa: E402
from tinyscribe.server import LlamaServer  # noqa: E402

FORMS = {
    "eci_form6": "ECI Form 6: new voter registration",
    "eci_form6a": "ECI Form 6A: overseas voter registration",
    "eci_form6b": "ECI Form 6B: Aadhaar for electoral roll",
    "eci_form7": "ECI Form 7: objection / deletion from roll",
    "eci_form8": "ECI Form 8: correction of entries",
    "pfrda_apy_reg": "Atal Pension Yojana registration",
    "rti_odisha_first_appeal": "RTI first appeal (Odisha)",
    "rti_odisha_second_appeal": "RTI second appeal (Odisha)",
    "synth_household": "Household survey",
    "synth_job_application": "Job application",
    "tn_icds_anganwadi_worker": "Anganwadi worker application (TN ICDS)",
    "wb_bsk_scheme_municipality": "Bangla Sahayata Kendra scheme form (WB)",
}
SCRIPTS = ["english", "hindi", "bengali", "tamil"]


def form_name(tid: str) -> str:
    base = tid.rsplit("_", 1)[0]
    return FORMS.get(base, base.replace("_", " "))


def schema_list() -> list[dict]:
    out = []
    for tid in template_ids():
        tpl = load_template(tid)
        lays = page_layouts(tpl)
        for lay in lays:
            out.append({"name": lay.schema_name, "form": form_name(tid), "language": tpl["language"],
                        "page": lay.page + 1, "pages": len(lays),
                        "items": len(lay.fields) + len(lay.checkboxes) + len(lay.tables)})
    return out


def sample_list(n_per: int = 1) -> dict[str, dict]:
    """A few held-out test pages per script, printed and handwritten, if data/ is present."""
    man = REPO / "data" / "manifest.jsonl"
    if not man.exists():
        return {}
    rows = [json.loads(x) for x in man.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "test" and r["degradation"] in ("clean", "light")]
    rng = random.Random(4)
    rng.shuffle(rows)
    picked = []
    for s in SCRIPTS:
        for hw in (False, True):
            picked += [r for r in rows if r["script"] == s and r["handwriting"] == hw][:n_per]
    return {r["id"]: {"id": r["id"], "schema": r["schema"], "script": r["script"], "handwriting": r["handwriting"],
                      "form": form_name(r["template_id"]), "image": r["image"], "target": r["target"]}
            for r in picked}


def rows_for(lay, data: dict, target: dict | None) -> list[dict]:
    """One row per item, in page order: printed label, value, and the ground truth when known."""
    data, t = data or {}, target or {}
    out = []

    def add(kind, key, label, val, truth):
        row = {"kind": kind, "key": key, "label": label, "value": val}
        if target is not None:
            row["truth"] = truth
            row["ok"] = normalize(val) == normalize(truth) if kind != "checkbox" else val == truth
        out.append(row)

    for f in lay.fields:
        add("field", f["id"], f["label_text"], data.get("fields", {}).get(f["id"], ""),
            t.get("fields", {}).get(f["id"], ""))
    for b in lay.checkboxes:
        add("checkbox", b["id"], b["label_text"], bool(data.get("checkboxes", {}).get(b["id"])),
            bool(t.get("checkboxes", {}).get(b["id"])))
    for tid, rows in lay.tables.items():
        for r in rows:
            val = data.get("tables", {}).get(tid, {}).get(f"r{r}", [])
            add("table", f"{tid}.r{r}", f"{tid}, row {r}", " | ".join(val),
                " | ".join(t.get("tables", {}).get(tid, {}).get(f"r{r}", [])))
    return out


class Demo:
    def __init__(self, server: LlamaServer):
        self.server = server
        self.schemas = schema_list()
        self.samples = sample_list()
        self.lock = threading.Lock()   # llama-server runs one slot: one page at a time

    def parse(self, req: dict) -> dict:
        sample = self.samples.get(req.get("sample") or "")
        if sample:
            image = (REPO / "data" / sample["image"]).read_bytes()
        else:
            b64 = req["image"].split(",", 1)[-1]
            image = base64.b64decode(b64)
        lay = page_layout(*parse_schema_name(req["schema"]))
        with self.lock:
            ex = extract(self.server.url, prepare(image), lay)
        target = sample["target"] if sample and sample["schema"] == req["schema"] else None
        rows = rows_for(lay, ex.data, target)
        res = {"data": ex.data, "seconds": round(ex.seconds, 1), "tokens": ex.completion_tokens, "rows": rows}
        if target is not None:
            res["correct"] = sum(r["ok"] for r in rows)
        return res


def handler(demo: Demo):
    index = (REPO / "app" / "index.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, index, "text/html; charset=utf-8")
            elif self.path == "/api/info":
                self.json({"model": demo.server.label, "schemas": demo.schemas,
                           "samples": [{k: v for k, v in s.items() if k != "target"}
                                       for s in demo.samples.values()]})
            elif self.path.startswith("/sample/"):
                s = demo.samples.get(self.path.removeprefix("/sample/"))
                if not s:
                    return self.send(404, b"not found", "text/plain")
                self.send(200, (REPO / "data" / s["image"]).read_bytes(), "image/jpeg")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/parse":
                return self.send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.json(demo.parse(req))
            except Exception as e:   # shown in the page, not a crash
                self.json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="9b", help="9b, 2b or a GGUF path")
    ap.add_argument("--mmproj")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    srv = LlamaServer(a.model, a.mmproj, cpu=a.cpu)
    # `kill` (SIGTERM) stops the demo like Ctrl+C, so llama-server is stopped with it
    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    print(f"loading {srv.label} ...", flush=True)
    with srv:
        demo = Demo(srv)
        http = ThreadingHTTPServer(("127.0.0.1", a.port), handler(demo))
        url = f"http://127.0.0.1:{a.port}"
        print(f"TinyScribe demo: {url}  ({len(demo.samples)} sample pages; Ctrl+C to stop)", flush=True)
        if not a.no_browser:
            webbrowser.open(url)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
