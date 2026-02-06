"""AlphaGenome Web Deployment - FastAPI Application."""

import logging
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.model_manager import model_manager
from app.schemas import (
    HealthResponse,
    IntervalPredictionRequest,
    PredictionResponse,
    SequencePredictionRequest,
    VariantPredictionRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlphaGenome Web API",
    description=(
        "Web API for Google DeepMind's AlphaGenome - a unified DNA sequence "
        "model for regulatory variant-effect prediction. Analyzes DNA sequences "
        "of up to 1 million base pairs at single base-pair resolution across "
        "diverse modalities including gene expression, splicing, chromatin "
        "features, and contact maps."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
async def startup_event():
    """Load the model on startup if configured."""
    if settings.preload_model:
        logger.info("Starting model preload in background thread...")
        thread = threading.Thread(target=model_manager.load_model, daemon=True)
        thread.start()
    else:
        logger.info(
            "Model preloading disabled. Use POST /api/load-model to load manually."
        )


# ─── Web UI ───────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ─── Health & Status ──────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        model_loading=model_manager.is_loading,
        model_error=model_manager.error,
    )


@app.post("/api/load-model")
async def load_model():
    """Manually trigger model loading."""
    if model_manager.is_loaded:
        return {"message": "Model already loaded."}
    if model_manager.is_loading:
        return {"message": "Model is currently loading."}

    thread = threading.Thread(target=model_manager.load_model, daemon=True)
    thread.start()
    return {"message": "Model loading started."}


# ─── Prediction Endpoints ────────────────────────────────────────────────────


@app.post("/api/predict/interval", response_model=PredictionResponse)
async def predict_interval(request: IntervalPredictionRequest):
    """Predict genomic tracks for a given interval.

    This endpoint predicts functional genomic signals (read coverage) for a
    specified genomic interval. The interval should be up to 1 Mb (1,048,576 bp).
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Check /api/health for status.",
        )

    interval_size = request.end - request.start
    if interval_size > settings.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Interval size ({interval_size:,} bp) exceeds maximum "
                f"({settings.max_sequence_length:,} bp)."
            ),
        )

    try:
        data = model_manager.predict_interval(
            chromosome=request.chromosome,
            start=request.start,
            end=request.end,
            organism=request.organism,
            output_types=request.output_types,
            ontology_terms=request.ontology_terms,
        )
        return PredictionResponse(success=True, data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/predict/variant", response_model=PredictionResponse)
async def predict_variant(request: VariantPredictionRequest):
    """Predict the effect of a genetic variant.

    This endpoint compares model predictions for reference and alternate
    alleles to assess the functional impact of a genetic variant.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Check /api/health for status.",
        )

    interval_size = request.end - request.start
    if interval_size > settings.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Interval size ({interval_size:,} bp) exceeds maximum "
                f"({settings.max_sequence_length:,} bp)."
            ),
        )

    try:
        data = model_manager.predict_variant(
            chromosome=request.chromosome,
            start=request.start,
            end=request.end,
            variant_position=request.variant_position,
            ref_bases=request.reference_bases,
            alt_bases=request.alternate_bases,
            organism=request.organism,
            output_types=request.output_types,
            ontology_terms=request.ontology_terms,
        )
        return PredictionResponse(success=True, data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Variant prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Variant prediction failed: {e}"
        )


@app.post("/api/predict/sequence", response_model=PredictionResponse)
async def predict_sequence(request: SequencePredictionRequest):
    """Predict genomic tracks from a raw DNA sequence.

    Provide a raw DNA sequence (A, C, G, T characters) and receive predictions
    for the requested output modalities.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Check /api/health for status.",
        )

    if len(request.sequence) > settings.max_sequence_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sequence length ({len(request.sequence):,} bp) exceeds "
                f"maximum ({settings.max_sequence_length:,} bp)."
            ),
        )

    try:
        data = model_manager.predict_sequence(
            sequence=request.sequence,
            organism=request.organism,
            output_types=request.output_types,
            ontology_terms=request.ontology_terms,
        )
        return PredictionResponse(success=True, data=data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Sequence prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Sequence prediction failed: {e}"
        )


# ─── Metadata Endpoints ──────────────────────────────────────────────────────


@app.get("/api/output-types")
async def get_output_types():
    """List all available output types."""
    return {
        "output_types": [
            {
                "id": "rna_seq",
                "name": "RNA-seq",
                "description": "RNA sequencing tracks capturing gene expression",
            },
            {
                "id": "cage",
                "name": "CAGE",
                "description": "Cap Analysis of Gene Expression tracks",
            },
            {
                "id": "dnase",
                "name": "DNase-seq",
                "description": "DNase I hypersensitive site tracks (chromatin accessibility)",
            },
            {
                "id": "atac",
                "name": "ATAC-seq",
                "description": "ATAC-seq tracks (chromatin accessibility)",
            },
            {
                "id": "chip_histone",
                "name": "Histone ChIP-seq",
                "description": "ChIP-seq tracks for histone modifications",
            },
            {
                "id": "chip_tf",
                "name": "TF ChIP-seq",
                "description": "ChIP-seq tracks for transcription factor binding",
            },
            {
                "id": "splice_sites",
                "name": "Splice Sites",
                "description": "Donor and acceptor splice site predictions",
            },
            {
                "id": "splice_site_usage",
                "name": "Splice Site Usage",
                "description": "Fraction of time each splice site is used",
            },
            {
                "id": "splice_junctions",
                "name": "Splice Junctions",
                "description": "Split read RNA-seq counts for each junction",
            },
            {
                "id": "contact_maps",
                "name": "Contact Maps",
                "description": "3D DNA-DNA contact probability maps",
            },
            {
                "id": "procap",
                "name": "PRO-cap",
                "description": "Precision Run-On sequencing and capping",
            },
        ]
    }


@app.get("/api/example-queries")
async def get_example_queries():
    """Return example queries for the web UI."""
    return {
        "interval_examples": [
            {
                "name": "TP53 Gene Region (chr17)",
                "chromosome": "chr17",
                "start": 7171710,
                "end": 8220286,
                "output_types": ["rna_seq"],
                "ontology_terms": [],
                "description": "Region around the TP53 tumor suppressor gene.",
            },
            {
                "name": "BRCA1 Region (chr17)",
                "chromosome": "chr17",
                "start": 43044295,
                "end": 44092871,
                "output_types": ["rna_seq", "dnase"],
                "ontology_terms": [],
                "description": "Region around the BRCA1 breast cancer gene.",
            },
        ],
        "variant_examples": [
            {
                "name": "Example SNV on chr22",
                "chromosome": "chr22",
                "start": 35677410,
                "end": 36725986,
                "variant_position": 36201698,
                "reference_bases": "A",
                "alternate_bases": "C",
                "output_types": ["rna_seq"],
                "ontology_terms": ["UBERON:0001157"],
                "description": (
                    "Single nucleotide variant on chromosome 22, predicting "
                    "RNA-seq in mouth mucosa tissue."
                ),
            },
        ],
    }
