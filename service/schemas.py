from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Interval(BaseModel):
    chromosome: str = Field(..., description="e.g. 'chr22'")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class Variant(BaseModel):
    chromosome: str = Field(..., description="e.g. 'chr22'")
    position: int = Field(..., ge=0, description="1-based genomic position")
    reference_bases: str = Field(..., min_length=1)
    alternate_bases: str = Field(..., min_length=1)


OutputMode = Literal["summary", "window"]
ChannelReduction = Literal["mean", "first", "none"]
Organism = Literal["human", "mouse"]


class PredictVariantRequest(BaseModel):
    interval: Interval
    variant: Variant
    organism: Organism = "human"
    ontology_terms: list[str] = Field(default_factory=list)
    requested_outputs: list[str] = Field(
        default_factory=lambda: ["RNA_SEQ"],
        description="Names from alphagenome_research.model.dna_model.OutputType (e.g. RNA_SEQ, DNASE, ATAC).",
    )

    output_mode: OutputMode = "summary"
    window_bp: int | None = Field(
        default=None,
        ge=1,
        description="If output_mode='window', return values from a window of this size around the variant.",
    )
    downsample_to: int | None = Field(
        default=None,
        ge=16,
        description="If output_mode='window', downsample to roughly this many points.",
    )
    channel_reduction: ChannelReduction = Field(
        default="mean",
        description="How to reduce multi-channel outputs when output_mode='window'.",
    )


class TrackSummary(BaseModel):
    shape: list[int]
    dtype: str
    min: float | None = None
    max: float | None = None
    mean: float | None = None


class TrackWindow(BaseModel):
    x_start: int
    x_end: int
    values: list[float]
    downsampled: bool
    channel_reduction: ChannelReduction


class PredictVariantResponse(BaseModel):
    backend: str
    model_handle: str | None = None
    organism: Organism
    requested_outputs: list[str]
    output_mode: OutputMode
    interval: Interval
    variant: Variant
    outputs: dict[str, dict[str, TrackSummary | TrackWindow]]

