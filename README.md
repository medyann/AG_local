# AlphaGenome API for Cursor Web

This repo provides a **small HTTP API** (FastAPI) that loads **AlphaGenome** from Hugging Face (`google/alphagenome-all-folds`) and exposes a `predict_variant` endpoint you can call from Cursor Web (or any client).

Model page: [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds)

## Important notes (gated model)

- **You must accept the model terms** on Hugging Face (the model is gated).
- You must provide a **Hugging Face access token** with at least **read** scope via `HF_TOKEN`.
- AlphaGenome inference is **heavy**. For real use you’ll want a **GPU machine**.

## Quickstart (mock backend, no token required)

The mock backend lets you validate wiring and Cursor integration without downloading the model.

```bash
python3 -m pip install -r requirements.txt
ALPHAGENOME_BACKEND=mock uvicorn service.main:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl -s http://127.0.0.1:8000/healthz | jq .
```

## Quickstart (real AlphaGenome backend)

1) Go to the model page and **accept the conditions**: [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds)

2) Create a Hugging Face token and export it:

```bash
export HF_TOKEN="hf_..."
```

3) Start the server:

```bash
python3 -m pip install -r requirements.txt
ALPHAGENOME_BACKEND=alphagenome uvicorn service.main:app --host 0.0.0.0 --port 8000
```

## Docker

Build:

```bash
docker build -t alphagenome-api .
```

Run (mock):

```bash
docker run --rm -p 8000:8000 -e ALPHAGENOME_BACKEND=mock alphagenome-api
```

Run (real; requires accepted terms + token):

```bash
docker run --rm -p 8000:8000 \
  -e ALPHAGENOME_BACKEND=alphagenome \
  -e HF_TOKEN="$HF_TOKEN" \
  alphagenome-api
```

## API usage

FastAPI exposes interactive docs at `GET /docs` and an OpenAPI spec at `GET /openapi.json`.

Example request (summary mode):

```bash
curl -s http://127.0.0.1:8000/predict_variant \
  -H 'content-type: application/json' \
  -d '{
    "interval": {"chromosome":"chr22","start":35677410,"end":36725986},
    "variant": {"chromosome":"chr22","position":36201698,"reference_bases":"A","alternate_bases":"C"},
    "ontology_terms": ["UBERON:0001157"],
    "requested_outputs": ["RNA_SEQ"],
    "output_mode": "summary"
  }' | jq .
```

If you want a small vector around the variant (downsampled), use `output_mode="window"`:

```bash
curl -s http://127.0.0.1:8000/predict_variant \
  -H 'content-type: application/json' \
  -d '{
    "interval": {"chromosome":"chr22","start":35677410,"end":36725986},
    "variant": {"chromosome":"chr22","position":36201698,"reference_bases":"A","alternate_bases":"C"},
    "requested_outputs": ["RNA_SEQ"],
    "output_mode": "window",
    "window_bp": 32768,
    "downsample_to": 1024,
    "channel_reduction": "mean"
  }' | jq .
```

## Using from Cursor Web

Two common options:

- **Call it from code** in your repo (Python/JS) using `fetch`/`requests` against your deployed URL.
- **(If available in your Cursor plan)** add it as an **OpenAPI tool** by pointing Cursor to your server’s OpenAPI spec (`/openapi.json`).

## Deployment tips

- Put this behind **HTTPS** (e.g. a managed load balancer or reverse proxy like nginx/Caddy).
- Run **one worker** per GPU (each worker loads the model). If you scale horizontally, each replica will download/load its own copy.
- Consider setting `HF_HOME` to a persistent disk so model weights are cached across restarts.

