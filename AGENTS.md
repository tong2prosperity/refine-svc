# Repository Guidelines

## Project Structure & Module Organization
- Root scripts drive workflows: `inference.py` / `inference_v2.py` handle conversion, `train.py` / `train_v2.py` cover fine-tuning, and `real-time-gui.py` launches the live UI.
- `modules/` contains model and DSP blocks (latest stack in `modules/v2/`); shared helpers live in `utils/`.
- All configuration YAMLs live under `configs/`, while reference assets and starter data live in `examples/`, `assets/`, and `data/`.
- The current regression check is `test_filemap.py`; keep additional tests colocated with the features they exercise.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` followed by `pip install -r requirements.txt` (use `requirements-mac.txt` on Apple Silicon).
- `python inference.py --source path/to/src.wav --target examples/reference/001.wav --output outputs/demo` validates zero-shot inference.
- `python real-time-gui.py --checkpoint-path checkpoints/seed-uvit.pth --config-path configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml` starts the streaming interface.
- `python train.py --config configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml --dataset-dir data/my_speaker --run-name my_run` runs fine-tuning.
- `python -m pytest test_filemap.py` (or `python test_filemap.py`) executes the smoke test; extend coverage as you add modules.
- `uvicorn svc_backend:app --host 0.0.0.0 --port 8000` boots the FastAPI backend exposing `GET /voices` and `WS /ws/convert?voice_id=<id>&sample_rate=16000` for real-time timbre conversion.

## Coding Style & Naming Conventions
- Target Python 3.10+, four-space indentation, snake_case identifiers, and PascalCase classes.
- Keep modules narrow in scope and mirror the existing package layout when adding new components.
- Maintain explicit device negotiation (`torch.device(...)` with CUDA/MPS fallbacks) and route file paths through CLI args or configs.

## Testing Guidelines
- Use `pytest`; name files `test_*.py` and rely on deterministic fixtures from `examples/` to keep runtime low.
- Cover both CPU code paths and GPU-conditional logic when relevant, mocking heavyweight downloads where possible.
- For training updates, assert on checkpoint creation or loss behavior so regressions surface quickly.

## Commit & Pull Request Guidelines
- Prefer short, present-tense commits consistent with history (e.g., `svc refine`, `fix dependency conflict`).
- PRs should explain behavior changes, note config or checkpoint updates, and list verification commands (`pytest`, sample inference, or training snippets).
- Attach brief audio samples or metrics when you touch conversion quality, and link tracking issues when available.

## Model Assets & Configuration Tips
- Checkpoints download automatically; set `HF_ENDPOINT=https://hf-mirror.com` if you need a mirror.
- Store custom weights under `checkpoints/` and reference them via CLI flags instead of hard-coding paths.
- Keep private datasets outside the repo and document new layouts with a lightweight `data/README.md` when needed.
