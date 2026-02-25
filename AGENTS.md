# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Single-file Python bioinformatics tool (`predict_and_trackhub.py`) that uses Google DeepMind's **AlphaGenome** model to predict functional genomic tracks for a mouse genome region (mm10 chr11:32,114,094–32,179,698) and exports results as a UCSC Track Hub (940 bigWig files).

### Python environment

- Python 3.12 virtual environment at `.venv/`
- Activate: `source /workspace/.venv/bin/activate`
- No `requirements.txt` or lockfile exists; dependencies are installed per README instructions

### Key dependencies

```bash
pip install git+https://github.com/google-deepmind/alphagenome_research.git
pip install pyBigWig
```

### Running the script

The prediction script requires **authentication credentials** to download model weights (gated but free):

- **HuggingFace** (recommended): set `HF_TOKEN` env var
- **Kaggle** (alternative): set `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars

```bash
source /workspace/.venv/bin/activate
python predict_and_trackhub.py --device cpu   # ~30-60 min on CPU
python predict_and_trackhub.py                # auto-detects GPU if available (~2 min)
```

### Linting

No project-level linting config exists. Use `ruff` (installed in venv):
```bash
ruff check predict_and_trackhub.py
```

### Testing

No automated test suite exists. Core functions can be validated by importing and calling them directly (see `compute_prediction_interval`, `sanitize_name`, `write_bigwig`, `create_track_hub`).

### Gotchas

- The `.bw` files in `trackhub_output/` are tracked via **Git LFS**. Run `git lfs pull` after clone to get actual data instead of pointer files.
- No GPU is available in Cloud Agent VMs; inference runs on CPU only (~30-60 min). The script includes a bfloat16 → float32 einsum patch for CPU compatibility.
- The CUDA error `Failed call to cuInit: UNKNOWN ERROR (303)` on import is benign — JAX falls back to CPU automatically.
- `python3.12-dev` must be installed as a system package for building C extensions (e.g., `sorted_nearest`).
- `python3.12-venv` must be installed for creating virtual environments.
