"""
Generate video/script.json and video/SCRIPT.md for the GlucoTwin explainer.

Timing rule from the brief: duration = words / 2.4 + 1 second of breathing room.
Keeping the scenes as data here means the runtime is computed, never guessed, and the
two output files can never drift apart.

Every factual claim below is checked against the repository:
  ablation numbers      artifacts/ablation.csv
  per-meal numbers      artifacts/meal_learning_summary.csv
  cohort / alignment    artifacts/metrics.json, artifacts/alignment_report.csv
  dashboard figures     read off the running app (docs/shots/)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
WPS = 2.4          # words per second
BREATH = 1.0       # seconds of breathing room per scene

# section, id, narration, visual, glimpse
from scenes import SCENES as S


def build():
    scenes, total = [], 0.0
    for section, sid, narration, visual, glimpse in S:
        words = len(narration.split())
        dur = round(words / WPS + BREATH, 1)
        total += dur
        scenes.append({
            "id": sid,
            "section": section,
            "durationSeconds": dur,
            "words": words,
            "narration": narration,
            "visual": visual,
            "glimpse": glimpse,
            "audio": f"public/vo/{sid}.mp3",
        })
    return scenes, total


def srt_time(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text, width=42, max_lines=2):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    # collapse to at most max_lines chunks, returned as a list of cue blocks
    return [lines[i:i + max_lines] for i in range(0, len(lines), max_lines)]


def main():
    scenes, total = build()
    (HERE / "script.json").write_text(json.dumps({
        "title": "GlucoTwin — explainer",
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "wordsPerSecond": WPS,
        "breathingRoomSeconds": BREATH,
        "totalSeconds": round(total, 1),
        "scenes": scenes,
    }, indent=2), encoding="utf-8")

    # ---- SCRIPT.md
    md = ["# GlucoTwin — explainer script\n",
          f"**{len(scenes)} scenes · {total/60:.1f} minutes · "
          f"{sum(s['words'] for s in scenes)} words**\n",
          "Timing: `words / 2.4 + 1s`. Replace a scene's estimate by dropping "
          "`public/vo/<id>.mp3` in and letting `calculateMetadata` size it to the audio.\n"]
    sec_order, seen = [], set()
    for s in scenes:
        if s["section"] not in seen:
            seen.add(s["section"]); sec_order.append(s["section"])
    md.append("## Runtime by section\n")
    md.append("| Section | Scenes | Minutes |")
    md.append("|---|---|---|")
    for sec in sec_order:
        rows = [s for s in scenes if s["section"] == sec]
        md.append(f"| {sec} | {len(rows)} | {sum(r['durationSeconds'] for r in rows)/60:.1f} |")
    md.append(f"| **total** | **{len(scenes)}** | **{total/60:.1f}** |\n")
    t = 0.0
    for sec in sec_order:
        md.append(f"\n## {sec}\n")
        for s in [x for x in scenes if x["section"] == sec]:
            md.append(f"**`{s['id']}`** · {s['durationSeconds']}s · "
                      f"glimpse: `{s['glimpse'] or '—'}` · starts {int(t//60)}:{int(t%60):02d}  ")
            md.append(f"> {s['narration']}\n")
            md.append(f"*Visual:* {s['visual']}\n")
            t += s["durationSeconds"]
    (HERE / "SCRIPT.md").write_text("\n".join(md), encoding="utf-8")

    # ---- subtitles.srt
    out, n, t = [], 1, 0.0
    for s in scenes:
        blocks = wrap(s["narration"])
        per = s["durationSeconds"] / max(len(blocks), 1)
        for bi, lines in enumerate(blocks):
            a, b = t + bi * per, t + (bi + 1) * per
            out.append(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n" + "\n".join(lines) + "\n")
            n += 1
        t += s["durationSeconds"]
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "subtitles.srt").write_text("\n".join(out), encoding="utf-8")

    glimpses = sorted({s["glimpse"] for s in scenes if s["glimpse"]})
    print(f"scenes      : {len(scenes)}")
    print(f"words       : {sum(s['words'] for s in scenes)}")
    print(f"runtime     : {total/60:.2f} min ({total:.0f}s)")
    print(f"scene range : {min(s['durationSeconds'] for s in scenes)}s - "
          f"{max(s['durationSeconds'] for s in scenes)}s")
    print(f"srt cues    : {n-1}")
    print(f"glimpses    : {len(glimpses)} -> {', '.join(glimpses)}")
    print("\nby section:")
    for sec in sec_order:
        rows = [s for s in scenes if s["section"] == sec]
        print(f"  {sec:<13} {len(rows):>2} scenes  {sum(r['durationSeconds'] for r in rows)/60:>5.2f} min")


if __name__ == "__main__":
    main()
