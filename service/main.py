from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from service.config import settings
from service.model_manager import AlphaGenomeNotConfigured, model_manager
from service.schemas import (
    PredictVariantRequest,
    PredictVariantResponse,
    TrackSummary,
    TrackWindow,
)


app = FastAPI(title="AlphaGenome API (Cursor Web)", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": settings.backend,
        "model_loaded": model_manager.loaded(),
        "model_handle": settings.hf_model_handle,
    }


@app.on_event("startup")
def _startup() -> None:
    if not settings.eager_load:
        return
    # Eager-load if configured; don't crash the whole server if token missing.
    try:
        model_manager.get()
    except Exception:
        return


def _to_numpy_values(track_obj: Any) -> np.ndarray:
    if track_obj is None:
        raise ValueError("track is None")
    values = getattr(track_obj, "values", track_obj)
    arr = np.asarray(values)
    return arr


def _summary(arr: np.ndarray) -> TrackSummary:
    # Only compute simple stats; avoid nan propagation surprises.
    if arr.size == 0:
        return TrackSummary(shape=list(arr.shape), dtype=str(arr.dtype))
    flat = arr.astype(np.float64).ravel()
    return TrackSummary(
        shape=list(arr.shape),
        dtype=str(arr.dtype),
        min=float(np.nanmin(flat)),
        max=float(np.nanmax(flat)),
        mean=float(np.nanmean(flat)),
    )


def _window_around_variant(
    *,
    arr: np.ndarray,
    interval_start: int,
    interval_end: int,
    variant_pos_1based: int,
    window_bp: int,
    downsample_to: int,
    channel_reduction: str,
) -> TrackWindow:
    if arr.ndim == 0:
        arr = arr.reshape((1,))

    interval_len = max(1, interval_end - interval_start)
    pos0 = max(interval_start, variant_pos_1based - 1)
    offset = int(np.clip(pos0 - interval_start, 0, interval_len - 1))

    # Map bp offset -> array index for arbitrary resolutions.
    n = int(arr.shape[0])
    center = int(round(offset / interval_len * max(1, n - 1)))
    half_bins = int(round((window_bp / interval_len) * n / 2))
    half_bins = max(1, half_bins)
    start = max(0, center - half_bins)
    end = min(n, center + half_bins)

    seg = arr[start:end]

    if seg.ndim == 2:
        if channel_reduction == "mean":
            seg = np.nanmean(seg.astype(np.float64), axis=1)
        elif channel_reduction == "first":
            seg = seg[:, 0].astype(np.float64)
        elif channel_reduction == "none":
            # Return list-of-lists (downsampled); handled below.
            pass
        else:
            raise ValueError(f"Unknown channel_reduction={channel_reduction}")
    elif seg.ndim > 2:
        raise ValueError(f"Unsupported track rank: {seg.ndim}")

    downsampled = False
    if seg.shape[0] > downsample_to:
        downsampled = True
        idx = np.linspace(0, seg.shape[0] - 1, num=downsample_to, dtype=int)
        seg = seg[idx]

    if seg.ndim == 2:
        # channel_reduction == "none"
        values_out: list[float] | list[list[float]] = seg.astype(np.float64).tolist()
        # Flatten for schema compatibility: keep only mean of channels.
        # (If you need full multi-channel output, extend the schema.)
        values_out = [float(np.nanmean(np.asarray(v))) for v in values_out]  # type: ignore[arg-type]
        channel_reduction = "mean"
    else:
        values_out = [float(x) for x in np.asarray(seg, dtype=np.float64).ravel()]

    return TrackWindow(
        x_start=int(start),
        x_end=int(end),
        values=values_out,  # type: ignore[arg-type]
        downsampled=downsampled,
        channel_reduction=channel_reduction,  # type: ignore[arg-type]
    )


@app.post("/predict_variant", response_model=PredictVariantResponse)
def predict_variant(req: PredictVariantRequest) -> PredictVariantResponse:
    try:
        loaded = model_manager.get()
    except AlphaGenomeNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Build alphagenome objects only if we are using the real backend.
    if loaded.backend == "alphagenome":
        from alphagenome.data import genome  # type: ignore
        from alphagenome_research.model import dna_model  # type: ignore

        interval = genome.Interval(
            chromosome=req.interval.chromosome,
            start=req.interval.start,
            end=req.interval.end,
        )
        variant = genome.Variant(
            chromosome=req.variant.chromosome,
            position=req.variant.position,
            reference_bases=req.variant.reference_bases,
            alternate_bases=req.variant.alternate_bases,
        )

        # Validate and map requested outputs.
        valid = {e.name: e for e in dna_model.OutputType}
        out_enums = []
        for name in req.requested_outputs:
            if name not in valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown requested_output '{name}'. Valid examples: {', '.join(list(valid)[:8])} ...",
                )
            out_enums.append(valid[name])

        kwargs: dict[str, Any] = dict(
            interval=interval,
            variant=variant,
            ontology_terms=req.ontology_terms,
            requested_outputs=out_enums,
        )
        # Organism enum names may differ across releases; best-effort.
        org = None
        if hasattr(dna_model, "Organism"):
            if req.organism == "human":
                org = getattr(dna_model.Organism, "HOMO_SAPIENS", None)
            else:
                org = getattr(dna_model.Organism, "MUS_MUSCULUS", None)
        if org is not None:
            kwargs["organism"] = org

        outputs = loaded.model.predict_variant(**kwargs)
        model_handle = loaded.handle
    else:
        # Mock backend.
        outputs = loaded.model.predict_variant(
            interval=req.interval.model_dump(),
            variant=req.variant.model_dump(),
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
        )
        model_handle = None

    window_bp = req.window_bp or settings.default_window_bp
    downsample_to = req.downsample_to or settings.default_downsample_to

    per_output: dict[str, dict[str, TrackSummary | TrackWindow]] = {}

    for name in req.requested_outputs:
        attr = name.lower()
        ref_track = getattr(outputs.reference, attr, None)
        alt_track = getattr(outputs.alternate, attr, None)
        if ref_track is None or alt_track is None:
            continue

        try:
            ref_arr = _to_numpy_values(ref_track)
            alt_arr = _to_numpy_values(alt_track)
        except Exception:
            continue

        if req.output_mode == "summary":
            per_output[name] = {"reference": _summary(ref_arr), "alternate": _summary(alt_arr)}
        else:
            per_output[name] = {
                "reference": _window_around_variant(
                    arr=ref_arr,
                    interval_start=req.interval.start,
                    interval_end=req.interval.end,
                    variant_pos_1based=req.variant.position,
                    window_bp=window_bp,
                    downsample_to=downsample_to,
                    channel_reduction=req.channel_reduction,
                ),
                "alternate": _window_around_variant(
                    arr=alt_arr,
                    interval_start=req.interval.start,
                    interval_end=req.interval.end,
                    variant_pos_1based=req.variant.position,
                    window_bp=window_bp,
                    downsample_to=downsample_to,
                    channel_reduction=req.channel_reduction,
                ),
            }

    return PredictVariantResponse(
        backend=loaded.backend,
        model_handle=model_handle,
        organism=req.organism,
        requested_outputs=req.requested_outputs,
        output_mode=req.output_mode,
        interval=req.interval,
        variant=req.variant,
        outputs=per_output,
    )

