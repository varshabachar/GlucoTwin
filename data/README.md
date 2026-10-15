# Data

From the project folder run

    python scripts/download_data.py

It downloads the CORE files (~360 MB) straight from PhysioNet into `data/raw/big-ideas/`: glucose,
food logs, heart rate and beat-to-beat intervals for all 16 people. That is enough to run everything:
activity and sleep are estimated from heart rate when accelerometer files are absent.

The download is safe to stop and re-run; finished files are skipped and partial ones resume.

Optional (large, slow): `--extra` adds skin temperature + EDA (~2.6 GB), `--acc` adds the
accelerometer (~13 GB). After adding files, rebuild with `python run_pipeline.py --no-cache`.

Raw data is never committed (see `.gitignore`).

No download yet? `python scripts/make_demo_data.py` creates lookalike files in `data/demo/`
(same format, simulated values) and `python run_pipeline.py --demo` runs everything on them.
