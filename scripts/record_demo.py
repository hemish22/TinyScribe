#!/usr/bin/env python3
"""Record the README GIF from the running web demo (headless Chromium via Playwright, then ffmpeg).

    python app/server.py --no-browser &          # 9B IQ3_XXS on http://127.0.0.1:8000
    uv run --with playwright python scripts/record_demo.py --sample "BN · handwritten"

The wait while the model reads the page is sped up in the GIF (--wait-speed, default 8x); the timer
in the page keeps showing real seconds, so the true speed stays visible.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def record(url: str, sample: str, video_dir: Path, size: tuple[int, int]) -> tuple[Path, float, float]:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1,
                                  record_video_dir=str(video_dir),
                                  record_video_size={"width": size[0], "height": size[1]},
                                  color_scheme="light")
        page = ctx.new_page()
        t0 = time.time()
        page.goto(url)
        page.wait_for_selector(".samples button")
        page.wait_for_timeout(1200)
        page.get_by_role("button", name=sample).first.click()
        page.wait_for_timeout(1800)
        page.click("#go")
        t_click = time.time() - t0
        page.wait_for_selector("#tabs:not([hidden])", timeout=600_000)
        t_done = time.time() - t0
        page.wait_for_timeout(2500)
        res = page.locator("#results")
        for _ in range(6):                              # scroll through the fields
            res.evaluate("el => el.scrollBy({top: 140, behavior: 'smooth'})")
            page.wait_for_timeout(700)
        page.wait_for_timeout(800)
        page.click("#tabs button[data-view=json]")
        page.wait_for_timeout(2500)
        page.click("#tabs button[data-view=table]")
        page.wait_for_timeout(1500)
        video = Path(page.video.path())
        ctx.close()
        browser.close()
    return video, t_click, t_done


def to_gif(video: Path, out: Path, t_click: float, t_done: float, speed: float, width: int, fps: int) -> None:
    a, b = t_click + 0.8, t_done - 0.3             # sped-up stretch: the model reading the page
    fl = (f"[0:v]trim=0:{a},setpts=PTS-STARTPTS[v0];"
          f"[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)/{speed}[v1];"
          f"[0:v]trim={b},setpts=PTS-STARTPTS[v2];"
          f"[v0][v1][v2]concat=n=3:v=1,fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
          f"[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", fl, str(out)],
                   check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--sample", default="BN · handwritten", help="sample button text in the demo")
    ap.add_argument("--out", default=str(REPO / "docs" / "demo.gif"))
    ap.add_argument("--wait-speed", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        video, t_click, t_done = record(a.url, a.sample, Path(tmp), (1280, 760))
        print(f"recorded {video.name}: clicked at {t_click:.1f}s, answer at {t_done:.1f}s")
        keep = REPO / "runs" / "demo_recording.webm"
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(video, keep)
        to_gif(keep, Path(a.out), t_click, t_done, a.wait_speed, a.width, a.fps)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
