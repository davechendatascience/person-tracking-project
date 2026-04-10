# Person tracking (BoxMOT)

Local setup for multi-object tracking experiments using **[BoxMOT](https://github.com/mikel-brostrom/boxmot)** on **MOT17** and **MOT20** ablation benchmarks (bounding-box MOT). This repository adds helper scripts for **Windows**, **managed corporate networks** (SSL inspection), and reliable **`uv`-based** optional dependencies.

## Requirements

- **Python 3.10+** (3.12 works; `truststore` needs 3.10+)
- **Git**
- **Windows**: PowerShell is used in the examples below.

## Quick start (new machine)

```powershell
cd person-tracking-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

BoxMOT’s default MOT17/MOT20 ablation configs use **YOLOX** detectors. Missing packages are installed at runtime via **`uv pip install`**, so **`uv` must be available** (it is listed in `requirements.txt`). If you call Python **without** activating the venv, use the launchers below so `uv` is on `PATH`.

### Run evaluation (recommended entry points)

Use UTF-8 mode on Windows (avoids encoding issues in some BoxMOT code paths) and the launcher that enables **truststore** (OS certificate store) plus **venv `Scripts` on `PATH`**:

```powershell
.\scripts\Invoke-Boxmot.ps1 eval --benchmark mot17-ablation --tracker boosttrack --verbose
.\scripts\Invoke-Boxmot.ps1 eval --benchmark mot20-ablation --tracker boosttrack --verbose
```

Equivalent manual invocation:

```powershell
$env:PYTHONUTF8 = "1"
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
.\.venv\Scripts\python.exe scripts\run_boxmot.py eval --benchmark mot17-ablation --tracker boosttrack --verbose
```

Optional postprocessing (see upstream docs): `--postprocessing gsi` or `--postprocessing gbrc`.

Precompute detections/embeddings once, then re-run `eval`:

```powershell
.\scripts\Invoke-Boxmot.ps1 generate --benchmark mot17-ablation
.\scripts\Invoke-Boxmot.ps1 generate --benchmark mot20-ablation
```

### Corporate network / SSL inspection

If HTTPS downloads fail with certificate errors:

1. Try **`truststore`** (uses the Windows certificate store). The **`scripts/run_boxmot.py`** launcher loads it automatically.
2. If that is not enough, merge your organization’s PEM with certifi and set env vars: **`scripts/Configure-PythonSSL.ps1`** (see comments inside the script).
3. Diagnose with **`scripts/test_https.py`**.

## Project layout

| Path | Purpose |
|------|---------|
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies (`boxmot`, `ultralytics`, `uv`, `truststore`, …) |
| [`scripts/run_boxmot.py`](scripts/run_boxmot.py) | Launch BoxMOT with `truststore` + `PATH` fix for `uv` |
| [`scripts/Invoke-Boxmot.ps1`](scripts/Invoke-Boxmot.ps1) | PowerShell wrapper: `PYTHONUTF8`, `PATH`, then `run_boxmot.py` |
| [`scripts/Configure-PythonSSL.ps1`](scripts/Configure-PythonSSL.ps1) | Build combined CA bundle for managed networks |
| [`scripts/merge_python_ssl_bundle.py`](scripts/merge_python_ssl_bundle.py) | Merge certifi + corporate PEM files |
| [`scripts/test_https.py`](scripts/test_https.py) | Quick HTTPS probe (PyPI, GitHub, Google) |
| [`ssl/`](ssl/) | Local CA bundles (gitignored); keep only what you need on each machine |

Dataset and run caches are **not** committed; BoxMOT downloads benchmarks under its package paths or project `runs/` as configured.

## Git: what is ignored

[`.gitignore`](.gitignore) is set up for Python, ML weights/archives, experiment outputs, caches, common IDEs (with **exceptions** so you can commit shared `.vscode/*.json` and `.cursor/rules/`), and secrets. See comments inside the file. Copy [`.env.example`](.env.example) to `.env` for optional local variables (`.env` stays untracked).

## Upstream documentation

- BoxMOT repo and CLI: [mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)
- Evaluation shortcuts and metrics are described in the upstream `README`.

## License

Dependencies (BoxMOT, PyTorch, etc.) have their own licenses. Add a project license file here if you redistribute this work.
