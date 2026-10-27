#!/usr/bin/env python3
"""Record the README GIF from the running web demo (headless Chrome via Playwright, then ffmpeg).

    python app/server.py --no-browser &          # 9B IQ3_XXS on http://127.0.0.1:8000; start it fresh:
                                                 # a page read before is answered faster from llama.cpp's
                                                 # slot cache, which would overstate the speed
    uv pip install playwright && python scripts/record_demo.py --sample "BN · handwritten"

Takes timed screenshots while it clicks through the demo and joins them with ffmpeg (no Playwright
video download needed). The wait while the model reads the page is sped up in the GIF
(--wait-speed, default 8x); the timer in the page keeps showing real seconds.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class Frames:
    def __init__(self, page, out: Path, fps: int):
        self.page, self.out, self.dt = page, out, 1 / fps
        self.items: list[tuple[Path, float]] = []

    def shot(self, duration: float) -> None:
        p = self.out / f"{len(self.items):05d}.png"
        self.page.screenshot(path=str(p))
        self.items.append((p, duration))

    def hold(self, seconds: float) -> None:
        """Real-time frames for `seconds` (screenshots take part of each frame's time)."""
        end = time.time() + seconds
        while time.time() < end:
            t = time.time()
            self.shot(self.dt)
            time.sleep(max(0.0, self.dt - (time.time() - t)))

    def concat_file(self) -> Path:
        f = self.out / "frames.txt"
        lines = [f"file '{p}'\nduration {d:.3f}" for p, d in self.items]
        lines.append(f"file '{self.items[-1][0]}'")   # ffmpeg concat needs the last frame twice
        f.write_text("\n".join(lines) + "\n")
        return f


def record(url: str, sample: str, tmp: Path, size: tuple[int, int], fps: int, speed: float) -> Frames:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:                      # Playwright's own Chromium not downloaded: use Chrome
            browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": size[0], "height": size[1]}, color_scheme="light")
        fr = Frames(page, tmp, fps)
        page.goto(url)
        page.wait_for_selector(".samples button")
        fr.hold(1.2)
        page.get_by_role("button", name=sample).first.click()
        page.wait_for_load_state("networkidle")
        fr.hold(1.8)
        page.click("#go")
        fr.hold(1.0)
        status = page.locator("#status")
        while not status.inner_text().startswith(("Done", "Error")):   # model reading: 1 frame per real s
            fr.shot(1.0 / speed)
            page.wait_for_timeout(1000)
            if "err" in (status.inner_html() or ""):
                raise SystemExit(f"demo error: {status.inner_text()}")
        fr.hold(2.5)
        for _ in range(5):                          # scroll through the fields
            page.mouse.move(size[0] * 0.75, size[1] * 0.6)
            page.mouse.wheel(0, 160)
            fr.hold(0.7)
        fr.hold(0.8)
        page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        fr.hold(0.8)
        page.click("#tabs button[data-view=json]")
        fr.hold(2.5)
        page.click("#tabs button[data-view=table]")
        fr.hold(1.5)
        browser.close()
    return fr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--sample", default="BN · handwritten", help="sample button text in the demo")
    ap.add_argument("--out", default=str(REPO / "docs" / "demo.gif"))
    ap.add_argument("--wait-speed", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        fr = record(a.url, a.sample, Path(tmp), (1280, 760), a.fps, a.wait_speed)
        print(f"{len(fr.items)} frames, {sum(d for _, d in fr.items):.1f} s of GIF")
        fl = (f"fps={a.fps},scale={a.width}:-1:flags=lanczos,split[s0][s1];"
              "[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(fr.concat_file()), "-filter_complex", fl, a.out], check=True)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
