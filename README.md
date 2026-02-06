# AlphaGenome Web Deployment

A web interface and REST API for [Google DeepMind's AlphaGenome](https://huggingface.co/google/alphagenome-all-folds) — a unified DNA sequence model for regulatory variant-effect prediction. Analyzes DNA sequences of up to 1 million base pairs at single base-pair resolution across diverse modalities.

## Features

- **Web UI** — Interactive browser interface with genomic track visualization
- **REST API** — Full-featured API with OpenAPI/Swagger documentation at `/docs`
- **Three prediction modes:**
  - **Interval Prediction** — Predict functional genomic signals for a chromosomal region
  - **Variant Effect** — Compare reference vs. alternate alleles to assess variant impact
  - **Sequence Prediction** — Predict directly from raw DNA sequence input
- **11 output modalities** — RNA-seq, CAGE, DNase-seq, ATAC-seq, histone ChIP-seq, TF ChIP-seq, splice sites, splice site usage, splice junctions, contact maps, PRO-cap
- **Docker support** — CPU and GPU (NVIDIA CUDA) deployment options

## Prerequisites

1. **Hugging Face account** — Create one at [huggingface.co](https://huggingface.co)
2. **Accept model terms** — Visit [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) and accept the license
3. **HF API token** — Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Hardware** — The model requires significant RAM (~32 GB+). GPU recommended for faster inference.

## Quick Start

### Option 1: Run Locally

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# 4. Start the server
python run.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Option 2: Docker (CPU)

```bash
# 1. Copy and edit environment file
cp .env.example .env
# Edit .env and set your HF_TOKEN

# 2. Build and run
docker compose up --build
```

### Option 3: Docker with GPU (NVIDIA)

```bash
# 1. Copy and edit environment file
cp .env.example .env
# Edit .env and set your HF_TOKEN

# 2. Build and run with GPU profile
docker compose --profile gpu up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/docs` | Interactive API documentation (Swagger) |
| `GET` | `/api/health` | Health check and model status |
| `POST` | `/api/load-model` | Manually trigger model loading |
| `POST` | `/api/predict/interval` | Predict for a genomic interval |
| `POST` | `/api/predict/variant` | Predict variant effects |
| `POST` | `/api/predict/sequence` | Predict from raw DNA sequence |
| `GET` | `/api/output-types` | List available output types |
| `GET` | `/api/example-queries` | Get example queries |

### Example API Call

```bash
# Variant effect prediction
curl -X POST http://localhost:8000/api/predict/variant \
  -H "Content-Type: application/json" \
  -d '{
    "chromosome": "chr22",
    "start": 35677410,
    "end": 36725986,
    "variant_position": 36201698,
    "reference_bases": "A",
    "alternate_bases": "C",
    "output_types": ["rna_seq"],
    "ontology_terms": ["UBERON:0001157"]
  }'
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HF_TOKEN` | *(required)* | Hugging Face API token |
| `ALPHAGENOME_MODEL_VERSION` | `all_folds` | Model version to load |
| `PRELOAD_MODEL` | `true` | Load model on server startup |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration and settings
│   ├── main.py             # FastAPI application and routes
│   ├── model_manager.py    # AlphaGenome model loading and inference
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── static/
│   │   ├── app.js          # Frontend JavaScript
│   │   └── style.css       # Frontend styles
│   └── templates/
│       └── index.html      # Web UI template
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # CPU Docker image
├── Dockerfile.gpu          # GPU Docker image (NVIDIA CUDA)
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
├── run.py                  # Application entry point
└── README.md
```

## Model Information

AlphaGenome is a unified DNA sequence model from Google DeepMind that:

- Processes up to **1 Mb** (1,048,576 bp) DNA sequences
- Delivers predictions at **single base-pair resolution**
- Covers **11 diverse modalities** (expression, splicing, chromatin, contact maps)
- Achieves **state-of-the-art** on 22/24 genome track benchmarks and 25/26 variant effect benchmarks

For more details, see the [AlphaGenome paper](https://doi.org/10.1038/s41586-025-10014-0) and [Hugging Face model card](https://huggingface.co/google/alphagenome-all-folds).

## License

- Application code: MIT License
- AlphaGenome model weights: [Non-commercial use only](https://huggingface.co/google/alphagenome-all-folds) (Google DeepMind terms)
