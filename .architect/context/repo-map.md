# Repository map

- Purpose: Hugging Face GLM-5.3-Flash model snapshot plus a production-ready
  H1-style harness packaged as `glmharness`.
- Runtime: Python 3.11+; optional PyTorch/Transformers for local inference
  via the `[inference]` extra.
- Package version: `0.4.0`.
- Entrypoint: `glm-harness` console script (also via `python3 harness.py`
  shim or `python3 -m glmharness.cli`); tests in `tests/`.
- Data: model shards and tokenizer/configuration files are vendored model
  artifacts; do not edit.
- Validation:
  - `python3 -m pytest -q` — full unit-test suite (integration tests are opt-in).
  - `python3 -m ruff check src tests` — lint.
  - `python3 harness.py --mock 'ok' 'ping'` — smoke run (shim path).
  - `glm-harness 'explain this repo'` — live smoke run (requires model).
