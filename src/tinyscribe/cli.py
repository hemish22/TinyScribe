"""tinyscribe: read a filled Indian form page into JSON, offline.

    tinyscribe parse form.jpg --schema eci_form7_hi_p0
    tinyscribe parse page1.jpg page2.jpg --schema synth_job_application_ta_p0 --model 2b
    tinyscribe parse form.jpg --schema eci_form7_hi_p0 --labels       # label: value, one per line
    tinyscribe schemas --language hindi

Starts llama-server with the chosen model, reads each image and stops the server (or uses a running
one with --server). Output is the JSON the model wrote, constrained by the page's JSON Schema.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from tinyscribe.client import extract
from tinyscribe.image import prepare
from tinyscribe.schema import load_template, page_layout, page_layouts, parse_schema_name, template_ids

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n")[0])


def schema_names() -> list[str]:
    return [lay.schema_name for tid in template_ids() for lay in page_layouts(load_template(tid))]


def labeled(lay, data: dict) -> list[tuple[str, str]]:
    """(printed label, value) pairs in page order."""
    out = [(f["label_text"], data.get("fields", {}).get(f["id"], "")) for f in lay.fields]
    out += [(b["label_text"], "yes" if data.get("checkboxes", {}).get(b["id"]) else "no")
            for b in lay.checkboxes]
    for tid, rows in data.get("tables", {}).items():
        for r, cells in rows.items():
            out.append((f"{tid} {r}", " | ".join(cells)))
    return out


def short(label: str, n: int = 48) -> str:
    label = " ".join(label.split())
    return label if len(label) <= n else label[: n - 1] + "…"


@app.command()
def parse(
    images: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="page image(s): JPEG, PNG, ..."),
    schema: str = typer.Option(..., "--schema", "-s", help="form page, e.g. eci_form7_hi_p0 (see `tinyscribe schemas`)"),
    model: str = typer.Option("9b", "--model", "-m", help="9b, 2b, or a GGUF path"),
    mmproj: Optional[str] = typer.Option(None, help="vision projector GGUF (for a model path)"),
    server: Optional[str] = typer.Option(None, help="use a running llama-server instead, e.g. http://127.0.0.1:8081"),
    cpu: bool = typer.Option(False, help="CPU only (no Metal/CUDA); about 3x slower"),
    labels: bool = typer.Option(False, "--labels", "-l", help="print 'label: value' lines instead of JSON"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="write JSON here (JSON Lines for several images)"),
):
    """Read form page images into JSON."""
    names = schema_names()
    if schema not in names:
        close = difflib.get_close_matches(schema, names, n=5, cutoff=0.5)
        typer.echo(f"unknown schema '{schema}'" + (f"; did you mean: {', '.join(close)}" if close else ""), err=True)
        raise typer.Exit(2)
    lay = page_layout(*parse_schema_name(schema))

    def run(url: str) -> list[dict]:
        results = []
        for img in images:
            ex = extract(url, prepare(img), lay)
            typer.echo(f"{img.name}: {ex.seconds:.1f} s, {ex.completion_tokens} tokens"
                       + ("" if ex.data is not None else " (output did not parse)"), err=True)
            results.append({"image": str(img), "schema": schema, "data": ex.data, "seconds": round(ex.seconds, 2)})
        return results

    if server:
        results = run(server)
    else:
        from tinyscribe.server import LlamaServer
        srv = LlamaServer(model, mmproj, cpu=cpu)
        typer.echo(f"loading {srv.label} ...", err=True)
        with srv:
            results = run(srv.url)

    if labels:
        for r in results:
            if len(results) > 1:
                typer.echo(f"== {r['image']}")
            pairs = [(short(lab), val) for lab, val in labeled(lay, r["data"] or {})]
            width = max((len(lab) for lab, _ in pairs), default=0)
            for lab, val in pairs:
                typer.echo(f"{lab:<{width}}  {val}")
    text = (json.dumps(results[0]["data"], ensure_ascii=False, indent=2) if len(results) == 1
            else "\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    if out:
        out.write_text(text + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)
    elif not labels:
        typer.echo(text)
    if any(r["data"] is None for r in results):
        raise typer.Exit(1)


@app.command()
def schemas(language: Optional[str] = typer.Option(None, help="english, hindi, bengali or tamil")):
    """List the form pages TinyScribe can read."""
    for tid in template_ids():
        tpl = load_template(tid)
        if language and tpl["language"] != language:
            continue
        for lay in page_layouts(tpl):
            n = len(lay.fields), len(lay.checkboxes), len(lay.tables)
            typer.echo(f"{lay.schema_name:<42} {tpl['language']:<8} "
                       f"{n[0]:>3} fields {n[1]:>3} checkboxes {n[2]:>2} tables")


def main() -> None:   # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":
    main()
