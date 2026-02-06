"""Pydantic schemas for API request/response models."""

from typing import Optional
from pydantic import BaseModel, Field


class IntervalPredictionRequest(BaseModel):
    """Request schema for interval-based predictions."""

    chromosome: str = Field(
        ...,
        description="Chromosome name (e.g., 'chr22')",
        examples=["chr22"],
    )
    start: int = Field(
        ...,
        description="Start position of the genomic interval",
        examples=[35677410],
    )
    end: int = Field(
        ...,
        description="End position of the genomic interval",
        examples=[36725986],
    )
    organism: str = Field(
        default="human",
        description="Organism type ('human' or 'mouse')",
    )
    output_types: list[str] = Field(
        default=["rna_seq"],
        description=(
            "List of output types to predict. Options: rna_seq, cage, dnase, "
            "atac, chip_histone, chip_tf, splice_sites, splice_site_usage, "
            "splice_junctions, contact_maps, procap"
        ),
    )
    ontology_terms: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of ontology terms to filter predictions "
            "(e.g., ['UBERON:0001157'] for mouth mucosa)"
        ),
    )


class VariantPredictionRequest(BaseModel):
    """Request schema for variant effect predictions."""

    chromosome: str = Field(
        ...,
        description="Chromosome name (e.g., 'chr22')",
        examples=["chr22"],
    )
    start: int = Field(
        ...,
        description="Start position of the genomic interval",
        examples=[35677410],
    )
    end: int = Field(
        ...,
        description="End position of the genomic interval",
        examples=[36725986],
    )
    variant_position: int = Field(
        ...,
        description="Position of the variant",
        examples=[36201698],
    )
    reference_bases: str = Field(
        ...,
        description="Reference base(s)",
        examples=["A"],
    )
    alternate_bases: str = Field(
        ...,
        description="Alternate base(s)",
        examples=["C"],
    )
    organism: str = Field(
        default="human",
        description="Organism type ('human' or 'mouse')",
    )
    output_types: list[str] = Field(
        default=["rna_seq"],
        description=(
            "List of output types to predict. Options: rna_seq, cage, dnase, "
            "atac, chip_histone, chip_tf, splice_sites, splice_site_usage, "
            "splice_junctions, contact_maps, procap"
        ),
    )
    ontology_terms: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of ontology terms to filter predictions "
            "(e.g., ['UBERON:0001157'])"
        ),
    )


class SequencePredictionRequest(BaseModel):
    """Request schema for raw sequence predictions."""

    sequence: str = Field(
        ...,
        description="Raw DNA sequence (A, C, G, T characters)",
        min_length=1,
    )
    organism: str = Field(
        default="human",
        description="Organism type ('human' or 'mouse')",
    )
    output_types: list[str] = Field(
        default=["rna_seq"],
        description=(
            "List of output types to predict. Options: rna_seq, cage, dnase, "
            "atac, chip_histone, chip_tf, splice_sites, splice_site_usage, "
            "splice_junctions, contact_maps, procap"
        ),
    )
    ontology_terms: Optional[list[str]] = Field(
        default=None,
        description="List of ontology terms to filter predictions",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_loading: bool
    model_error: Optional[str] = None


class PredictionResponse(BaseModel):
    """Generic prediction response wrapper."""

    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
