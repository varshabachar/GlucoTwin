"""
Record the GlucoTwin live demo, timed to the 8 beats of the demo script.

Each beat holds for exactly its scripted length, so the narration can be laid straight on top
without re-timing. A fake cursor is injected because Playwright recordings do not capture the
real pointer.

Output: brag-output/live-demo.mp4 (1920x1080), plus a per-beat timing table printed at the end.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(r"C:\Users\AMOG\Desktop\collage\sem 5\glucotwin\GlucoTwin_final\GlucoTwin")
OUT = ROOT / "brag-output"
RAW = OUT / "work" / "demo-raw"
RAW.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"
W, H = 1920, 1080

BEATS = [("C1", "The ward", 45), ("C2", "Split the ward", 20), ("C3", "One patient", 40),
         ("C4", "Plan a meal", 70), ("C5", "Any food", 35), ("C6", "The twin learning", 40),
         ("C7", "Women's module", 45), ("C8", "The numbers", 25)]

CURSOR = """
(() => {
  // The init script runs at document-start, when documentElement can still be null, so the
  // element is created lazily and every step is guarded. Listener goes on window first.
  let d = null;
  const ensure = () => {
    try {
      if (d && d.isConnected) return d;
      const host = document.body || document.documentElement;
      if (!host) return null;
      d = document.getElementById('__cur');
      if (!d) {
        d = document.createElement('div');
        d.id = '__cur';
        d.style.cssText = 'position:fixed;z-index:2147483647;width:24px;height:24px;'
          + 'margin:-12px 0 0 -12px;border-radius:50%;background:rgba(235,104,52,.95);'
          + 'box-shadow:0 0 0 4px rgba(235,104,52,.25),0 2px 10px rgba(0,0,0,.4);'
          + 'pointer-events:none;left:-99px;top:-99px;transition:transform .08s ease';
      }
      host.appendChild(d);
      return d;
    } catch (e) { return null; }
  };
  const put = (x, y) => { const el = ensure(); if (el) { el.style.left = x + 'px'; el.style.top = y + 'px'; } };
  addEventListener('mousemove', e => put(e.clientX, e.clientY), true);
  addEventListener('mousedown', () => { const el = ensure(); if (el) el.style.transform = 'scale(.55)'; }, true);
  addEventListener('mouseup',   () => { const el = ensure(); if (el) el.style.transform = 'scale(1)'; }, true);
  addEventListener('DOMContentLoaded', ensure);
  // Streamlit re-renders constantly; keep re-attaching
  setInterval(ensure, 500);
})();
"""


def settle(p, ms=2200):
    p.wait_for_timeout(ms)
    try:
        p.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=9000)
    except Exception:
        pass
    p.wait_for_timeout(500)


def nudge(p, thumb, key, n):
    thumb.focus()
    for _ in range(n):
        p.keyboard.press(key)
        p.wait_for_timeout(40)
    settle(p)


def glide(p, x, y, steps=22):
    """Move the fake cursor smoothly so the eye can follow it."""
    p.mouse.move(x, y, steps=steps)
    p.wait_for_timeout(260)


def point_at(p, locator, dwell=900):
    try:
        b = locator.first.bounding_box()
        if b:
            glide(p, b["x"] + b["width"] / 2, b["y"] + min(b["height"] / 2, 40))
            p.wait_for_timeout(dwell)
            return True
    except Exception:
        pass
    return False


def main():
    log = []
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        ctx = b.new_context(viewport={"width": W, "height": H},
                            record_video_dir=str(RAW),
                            record_video_size={"width": W, "height": H})
        ctx.add_init_script(CURSOR)
        pg = ctx.new_page()
        t0 = time.time()
        pg.goto(URL, wait_until="networkidle")
        settle(pg, 7000)

        # ---- setup before the clock starts: ward at day 3, 20:00
        sb = pg.get_by_test_id("stSidebar")
        nudge(pg, sb.get_by_role("slider").first, "ArrowRight", 1)
        nudge(pg, sb.get_by_role("slider").nth(1), "ArrowRight", 28)
        settle(pg, 2500)
        trim_from = time.time() - t0            # everything before this gets cut
        base = time.time()

        def hold(idx):
            """Wait until this beat's scripted end."""
            end = base + sum(x[2] for x in BEATS[:idx + 1])
            while time.time() < end:
                pg.wait_for_timeout(200)

        def mark(idx, note=""):
            log.append((BEATS[idx][0], BEATS[idx][1], BEATS[idx][2],
                        round(sum(x[2] for x in BEATS[:idx]), 1), note))

        # ---------------- C1  the ward
        try:
            point_at(pg, pg.get_by_text("Alerts now", exact=False))
            point_at(pg, pg.get_by_text("P-009", exact=True))
            glide(pg, 1500, 500)
            pg.wait_for_timeout(1200)
            point_at(pg, pg.get_by_text("Twin-projected glucose in 1 h", exact=False), 2500)
            mark(0)
        except Exception as e:
            mark(0, f"partial: {type(e).__name__}")
        hold(0)

        # ---------------- C2  cohort filter
        try:
            pg.get_by_text("Women 9", exact=True).click()
            settle(pg, 2200)
            point_at(pg, pg.get_by_text("post-menopausal", exact=False), 2000)
            mark(1)
        except Exception as e:
            mark(1, f"partial: {type(e).__name__}")
        hold(1)

        # ---------------- C3  one patient
        try:
            sb.get_by_text("Patient twin", exact=True).click(); settle(pg)
            pg.get_by_test_id("stSelectbox").first.click(); pg.wait_for_timeout(600)
            pg.get_by_text("P-005", exact=True).last.click(); settle(pg, 2600)
            nudge(pg, pg.get_by_test_id("stSidebar").get_by_role("slider").nth(1), "ArrowLeft", 16)
            settle(pg, 2200)
            pg.mouse.wheel(0, 900); pg.wait_for_timeout(1600)
            pg.mouse.wheel(0, 900); pg.wait_for_timeout(1600)
            mark(2)
        except Exception as e:
            mark(2, f"partial: {type(e).__name__}")
        hold(2)

        # ---------------- C4  plan a meal
        try:
            pg.get_by_text("Plan a meal", exact=False).first.scroll_into_view_if_needed()
            pg.wait_for_timeout(1500)
            peaks = pg.get_by_text("Projected peak", exact=False).last
            peaks.scroll_into_view_if_needed(); pg.mouse.wheel(0, -180); pg.wait_for_timeout(1500)
            point_at(pg, pg.get_by_text("Carbs first:", exact=False), 3000)
            point_at(pg, pg.get_by_text("Vegetables & protein first:", exact=False), 3500)
            point_at(pg, pg.get_by_text("Order effect:", exact=False), 3000)
            pg.mouse.wheel(0, 260); pg.wait_for_timeout(1200)
            point_at(pg, pg.get_by_text("Largest safe portion", exact=False), 3000)
            mark(3)
        except Exception as e:
            mark(3, f"partial: {type(e).__name__}")
        hold(3)

        # ---------------- C5  build your own / describe
        try:
            pg.get_by_text("Build your own", exact=True).click(); settle(pg, 2200)
            ci = pg.get_by_test_id("stNumberInput").first.locator("input")
            ci.click(); ci.fill(""); ci.type("120", delay=180); pg.keyboard.press("Enter")
            settle(pg, 2200)
            pg.get_by_text("Describe it in words", exact=True).click(); settle(pg, 2000)
            ti = pg.get_by_test_id("stTextInput").first.locator("input")
            ti.click(); ti.type("2 rotis, dal and a bowl of curd", delay=55)
            pg.wait_for_timeout(700)
            pg.get_by_role("button", name="Estimate nutrients").click()
            settle(pg, 5000)
            mark(4)
        except Exception as e:
            mark(4, f"partial: {type(e).__name__}")
        hold(4)

        # ---------------- C6  after the meal
        try:
            pg.get_by_text("After the meal", exact=False).first.scroll_into_view_if_needed()
            pg.wait_for_timeout(1600)
            sl = pg.get_by_test_id("stSidebar").page.get_by_role("slider")
            mealsl = sl.nth(sl.count() - 1)
            mealsl.focus()
            for _ in range(14):
                pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(120)
            settle(pg, 2600)
            point_at(pg, pg.get_by_text("Actual CGM peak", exact=False), 3500)
            mark(5)
        except Exception as e:
            mark(5, f"partial: {type(e).__name__}")
        hold(5)

        # ---------------- C7  women's module
        try:
            pg.get_by_text("Pick from the menu", exact=True).click(); settle(pg, 2200)
            pg.get_by_text("Cycle phase", exact=False).first.scroll_into_view_if_needed()
            pg.wait_for_timeout(1200)
            pg.get_by_text("Cycle phase", exact=False).first.click(); settle(pg, 2200)
            point_at(pg, pg.get_by_text("post-menopausal", exact=False), 4000)
            pg.get_by_text("Luteal (day 15-28) - demo", exact=False).first.click()
            settle(pg, 3000)
            peaks = pg.get_by_text("Projected peak", exact=False).last
            peaks.scroll_into_view_if_needed(); pg.mouse.wheel(0, -180); pg.wait_for_timeout(1400)
            point_at(pg, pg.get_by_text("Vegetables & protein first:", exact=False), 3500)
            mark(6)
        except Exception as e:
            mark(6, f"partial: {type(e).__name__}")
        hold(6)

        # ---------------- C8  model performance
        try:
            pg.get_by_test_id("stSidebar").get_by_text("Model performance", exact=True).click()
            settle(pg, 3000)
            pg.mouse.wheel(0, 700); pg.wait_for_timeout(2200)
            pg.mouse.wheel(0, 900); pg.wait_for_timeout(2200)
            mark(7)
        except Exception as e:
            mark(7, f"partial: {type(e).__name__}")
        hold(7)

        video = pg.video.path()
        ctx.close(); b.close()

    # ---- trim the setup off the front and convert to mp4
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = OUT / "live-demo.mp4"
    total = sum(x[2] for x in BEATS)
    r = subprocess.run([ff, "-y", "-ss", f"{trim_from:.2f}", "-i", str(video),
                        "-t", f"{total}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-preset", "medium", "-r", "30",
                        "-movflags", "+faststart", str(mp4)],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1200:]); raise SystemExit(1)

    print(f"\nwrote {mp4}  ({mp4.stat().st_size/1e6:.1f} MB)\n")
    print(f"{'beat':<5} {'what':<22} {'starts':>8} {'len':>5}  note")
    for bid, what, dur, start, note in log:
        print(f"{bid:<5} {what:<22} {int(start)//60}:{int(start)%60:02d}{'':>3} {dur:>4}s  {note}")
    print(f"\ntotal {total}s = {total//60}:{total%60:02d}")


if __name__ == "__main__":
    main()
