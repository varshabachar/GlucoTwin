"""
Make a small LOOKALIKE dataset in exactly the BIG IDEAs file format, so the whole pipeline and the
dashboard can be tested before (or without) the 4.7 GB real download.

    python scripts/make_demo_data.py            # 8 people x 5 days -> data/demo/

The glucose here is simulated from a known physiology (meals, activity, sleep), so results on demo data
mean nothing clinically - they only prove the code runs end to end. Use the real data for the submission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FOODS = [("Rice and dal", 1.0, "cup"), ("Oatmeal", 1.0, "cup"), ("Sandwich", 1.0, ""), ("Pasta", 1.5, "cup"),
         ("Apple", 1.0, ""), ("Yogurt", 1.0, "cup"), ("Roti with sabzi", 2.0, ""), ("Smoothie", 16.0, "fluid ounce")]


def _ts(arr) -> np.ndarray:
    return np.char.replace(np.datetime_as_string(arr, unit="us"), "T", " ")


def kernel(tau, tp):
    x = np.clip(tau, 0, None) / tp
    return np.where(tau > 0, x * np.exp(1 - x), 0.0)


def make_person(pid: str, sex: str, hba1c: float, days: int, start: pd.Timestamp, out: Path, rng):
    folder = out / pid
    folder.mkdir(parents=True, exist_ok=True)
    minutes = days * 1440
    t_min = np.arange(minutes, dtype=float)  # 1-minute simulation grid
    k = rng.uniform(0.6, 1.6) * (1 + 1.5 * (hba1c - 5.5))
    tp = rng.uniform(30, 70)
    a = rng.uniform(0.05, 0.25)
    base = 82 + 14 * (hba1c - 5.5) + rng.normal(0, 4)

    # ---- sleep (23:00-07:00 +/- jitter), some short nights
    asleep = np.zeros(minutes, bool)
    sleep_short = []
    for d in range(days):
        s = d * 1440 - 60 + int(rng.normal(0, 30))  # ~23:00 previous evening
        dur = int(rng.choice([300, 420, 480], p=[0.25, 0.45, 0.30]) + rng.normal(0, 20))
        lo, hi = max(s, 0), min(s + dur, minutes)
        asleep[lo:hi] = True
        sleep_short.append(dur < 360)

    # ---- meals
    meals = []
    for d in range(days):
        for (h0, h1, c0, c1) in [(7, 9, 25, 70), (12, 14, 40, 110), (18.5, 21, 40, 100)]:
            when = d * 1440 + rng.uniform(h0, h1) * 60
            meals.append((when, rng.uniform(c0, c1)))
        if rng.random() < 0.6:
            meals.append((d * 1440 + rng.uniform(15, 17) * 60, rng.uniform(10, 35)))
    meals.sort()

    # ---- activity (milli-g ENMO): low baseline, walks (often after meals), none during sleep
    act = rng.gamma(2.0, 6.0, minutes)
    for when, _ in meals:
        if rng.random() < 0.35:
            s = int(when + rng.uniform(10, 40))
            act[s:s + int(rng.uniform(10, 30))] += rng.uniform(120, 250)
    for d in range(days):
        if rng.random() < 0.5:
            s = int(d * 1440 + rng.uniform(6.5, 19) * 60)
            act[s:s + 40] += rng.uniform(150, 300)
    act[asleep] = rng.gamma(1.0, 2.0, asleep.sum())
    alpha = np.exp(-1 / config.TWIN_ACTIVITY_TAU)
    A = np.zeros(minutes)
    for i in range(1, minutes):
        A[i] = alpha * A[i - 1] + (1 - alpha) * act[i]

    # ---- glucose
    G = np.full(minutes, base)
    for when, c in meals:
        day = int(when // 1440)
        mult = 1.25 if sleep_short[min(day, days - 1)] else 1.0  # worse response after a short night
        G += mult * k * c * kernel(t_min - when, tp)
    G -= a * A
    hour = (t_min % 1440) / 60
    G += 8 * np.exp(-((hour - 6.5) ** 2) / 2)  # dawn phenomenon
    noise = np.zeros(minutes)
    for i in range(1, minutes):
        noise[i] = 0.995 * noise[i - 1] + rng.normal(0, 0.6)
    G += noise

    # ---- Dexcom file (every 5 min, odd seconds, a few gaps)
    cgm_idx = np.arange(int(rng.integers(0, 5)), minutes, 5)
    gap = rng.integers(0, len(cgm_idx) - 30)
    cgm_idx = np.delete(cgm_idx, np.arange(gap, gap + 12))
    secs = int(rng.integers(0, 60))
    times = start + pd.to_timedelta(cgm_idx, unit="m") + pd.Timedelta(seconds=secs)
    vals = np.round(G[cgm_idx] + rng.normal(0, 4, len(cgm_idx)))
    hdr = ("Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,Patient Info,Device Info,Source Device ID,"
           "Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams),Duration (hh:mm:ss),"
           "Glucose Rate of Change (mg/dL/min),Transmitter Time (Long Integer)")
    meta = [f"1,,FirstName,,2019,,,,,,,,", f"2,,LastName,,{pid},,,,,,,,", f"3,,PatientIdentifier,,2019-{pid},,,,,,,,",
            "4,,DateOfBirth,,1990-01-01,,,,,,,,", "5,,Device,,,Dexcom G6 Mobile App,iPhone G6,,,,,,",
            "6,,Alert,High,,,iPhone G6,200.0,,,,,"]
    lines = [hdr] + meta + [f"{i + 7},{t:%Y-%m-%d %H:%M:%S},EGV,,,,iPhone G6,{v:.1f},,,,,{11101 + i * 300}.0"
                            for i, (t, v) in enumerate(zip(times, vals))]
    (folder / f"Dexcom_{pid}.csv").write_text("\n".join(lines) + "\n")

    # ---- wristband: one hour off per day (charging)
    worn = np.ones(minutes, bool)
    for d in range(days):
        s = int(d * 1440 + rng.uniform(9, 20) * 60)
        worn[s:s + 60] = False

    def per_second(rate_hz: float):
        n = int(minutes * 60 * rate_hz)
        sec = np.arange(n) / rate_hz
        m = np.minimum((sec // 60).astype(int), minutes - 1)
        keep = worn[m]
        stamp = np.datetime64(start.to_datetime64(), "us") + (sec[keep] * 1e6).astype("timedelta64[us]")
        return stamp, m[keep]

    stamp, m = per_second(1.0)
    hr = 64 + 0.05 * act[m] - 7 * asleep[m] + rng.normal(0, 2, len(m))
    pd.DataFrame({"datetime": _ts(stamp), " hr": np.round(hr, 2)}).to_csv(folder / f"HR_{pid}.csv", index=False)
    acc_scale = 64.0
    vm = (1 + act[m] / 1000.0) * acc_scale
    theta = rng.uniform(0, np.pi, len(m))
    pd.DataFrame({"datetime": _ts(stamp), " acc_x": np.round(vm * np.cos(theta)),
                  " acc_y": np.round(vm * np.sin(theta) * 0.6), " acc_z": np.round(vm * np.sin(theta) * 0.8)}
                 ).to_csv(folder / f"ACC_{pid}.csv", index=False)
    stamp4, m4 = per_second(0.25)
    pd.DataFrame({"datetime": _ts(stamp4), " temp": np.round(33 - 1.5 * asleep[m4] + rng.normal(0, 0.2, len(m4)), 2)}
                 ).to_csv(folder / f"TEMP_{pid}.csv", index=False)
    pd.DataFrame({"datetime": _ts(stamp4), " eda": np.round(np.abs(0.4 + 0.002 * act[m4] + rng.normal(0, 0.1, len(m4))), 3)}
                 ).to_csv(folder / f"EDA_{pid}.csv", index=False)
    # IBI: beats in bursts (the E4 drops beats during movement)
    beats_t, beats = [], []
    for mm in range(0, minutes, 3):
        if not worn[mm] or act[mm] > 60:
            continue
        rr = 60 / (64 + 0.05 * act[mm] - 7 * asleep[mm])
        t = mm * 60.0
        for _ in range(int(rng.integers(5, 15))):
            ibi = max(rr + rng.normal(0, 0.04 + 0.02 * asleep[mm]), 0.35)
            t += ibi
            beats_t.append(t)
            beats.append(ibi)
    bt = np.datetime64(start.to_datetime64(), "us") + (np.array(beats_t) * 1e6).astype("timedelta64[us]")
    pd.DataFrame({"datetime": _ts(bt), " ibi": np.round(beats, 6)}).to_csv(folder / f"IBI_{pid}.csv", index=False)

    # ---- food log
    rows = []
    for when, c in meals:
        t = start + pd.Timedelta(minutes=float(when))
        name, amt, unit = FOODS[int(rng.integers(0, len(FOODS)))]
        fiber = round(c * rng.uniform(0.05, 0.15), 1)
        rows.append(dict(date=f"{t:%Y-%m-%d}", time=f"{t:%H:%M:%S}", time_begin=f"{t:%Y-%m-%d %H:%M:%S}",
                         time_end="", logged_food=name, amount=amt, unit=unit, searched_food=name,
                         calorie=round(c * 4 + rng.uniform(50, 300)), total_carb=round(c + fiber, 1),
                         dietary_fiber=fiber, sugar=round(c * rng.uniform(0.1, 0.5), 1),
                         protein=round(rng.uniform(5, 35), 1), total_fat=round(rng.uniform(3, 30), 1)))
    pd.DataFrame(rows).to_csv(folder / f"Food_Log_{pid}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=8)
    ap.add_argument("--days", type=int, default=5)
    args = ap.parse_args()
    out = config.DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    demo = []
    for i in range(1, args.people + 1):
        pid = f"{i:03d}"
        sex = "FEMALE" if rng.random() < 0.6 else "MALE"
        hba1c = round(float(rng.uniform(5.2, 6.4)), 1)
        demo.append(dict(ID=i, Gender=sex, HbA1c=hba1c))
        start = pd.Timestamp("2020-02-13 00:00:00") + pd.Timedelta(days=int(rng.integers(0, 60)))
        print(f"  making demo participant {pid} ...")
        make_person(pid, sex, hba1c, args.days, start, out, rng)
    pd.DataFrame(demo).to_csv(out / "Demographics.csv", index=False)
    print(f"Demo data written to {out}")


if __name__ == "__main__":
    main()
