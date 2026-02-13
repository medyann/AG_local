#!/usr/bin/env python3
"""AlphaGenome: Predict all modalities for mouse mm10 and create a UCSC Track Hub.

This script:
  1. Loads the AlphaGenome model (from HuggingFace or Kaggle).
  2. Predicts all genomic modalities for a 131,072 bp window centred on
     chr11:32,114,094-32,179,698 (mm10 / Mus musculus).
  3. Exports every non-padding track as a bigWig file.
  4. Assembles a UCSC Track Hub that can be loaded into the UCSC Genome Browser.

Predicted modalities (for mouse):
  - ATAC-seq           (18 tracks,  1 bp resolution)
  - DNase-seq          (67 tracks,  1 bp resolution)
  - CAGE               (188 tracks, 1 bp resolution, stranded)
  - RNA-seq            (173 tracks, 1 bp resolution, stranded)
  - ChIP-seq histone   (183 tracks, 128 bp resolution)
  - ChIP-seq TF        (127 tracks, 128 bp resolution)
  - Splice sites       (4 tracks,   1 bp resolution, donor/acceptor)
  - Splice-site usage  (180 tracks, 1 bp resolution, stranded)
  - Contact maps       (8 tracks,   2048 bp resolution, 2-D — skipped for bigWig)
  - PRO-cap            (0 tracks for mouse — nothing to export)

Usage
-----
  # With GPU (recommended – inference takes ~2 min):
  python predict_and_trackhub.py

  # Force CPU (slow but works – ~30–60 min on 16-core machine):
  python predict_and_trackhub.py --device cpu

  # Specify a public URL so the script prints a ready-to-click browser link:
  python predict_and_trackhub.py --hub-url https://myserver.com/trackhub_output

Requirements
------------
  pip install git+https://github.com/google-deepmind/alphagenome_research.git
  pip install pyBigWig

  For model download you need **one** of:
    * Kaggle credentials  (KAGGLE_USERNAME / KAGGLE_KEY env-vars, or ~/.kaggle/kaggle.json)
    * HuggingFace token   (huggingface-cli login)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make sure TensorFlow does not grab GPU memory (only used for data loading).
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import jax
import pyBigWig

from alphagenome.data import genome
from alphagenome.data import track_data as track_data_module
from alphagenome.models import dna_model, dna_output
from alphagenome_research.model import dna_model as ag_dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
TARGET_CHROM = "chr11"
TARGET_START = 32_114_094
TARGET_END   = 32_179_698
ORGANISM     = dna_model.Organism.MUS_MUSCULUS

# Standard AlphaGenome receptive field (2^17 bp).  The 65 604 bp target
# region fits comfortably inside this window.
MODEL_INPUT_LENGTH = 131_072

# Output types that produce 1-D positional tracks (suitable for bigWig).
BIGWIG_OUTPUT_TYPES = [
    dna_output.OutputType.ATAC,
    dna_output.OutputType.DNASE,
    dna_output.OutputType.CAGE,
    dna_output.OutputType.RNA_SEQ,
    dna_output.OutputType.CHIP_HISTONE,
    dna_output.OutputType.CHIP_TF,
    dna_output.OutputType.SPLICE_SITES,
    dna_output.OutputType.SPLICE_SITE_USAGE,
    dna_output.OutputType.PROCAP,
]

# Contact maps are 2-D matrices and cannot be represented as bigWig.
# Splice junctions are sparse pair-wise data – also not bigWig-compatible.
# Both are still *predicted* but not exported to the track hub.
ALL_OUTPUT_TYPES = BIGWIG_OUTPUT_TYPES + [
    dna_output.OutputType.CONTACT_MAPS,
    dna_output.OutputType.SPLICE_JUNCTIONS,
]

# mm10 chromosome sizes (GRCm38)
MM10_CHROM_SIZES: dict[str, int] = {
    "chr1": 195_471_971, "chr2": 182_113_224, "chr3": 160_039_680,
    "chr4": 156_508_116, "chr5": 151_834_684, "chr6": 149_736_546,
    "chr7": 145_441_459, "chr8": 129_401_213, "chr9": 124_595_110,
    "chr10": 130_694_993, "chr11": 122_082_543, "chr12": 120_129_022,
    "chr13": 120_421_639, "chr14": 124_902_244, "chr15": 104_043_685,
    "chr16": 98_207_768,  "chr17": 94_987_271,  "chr18": 90_702_639,
    "chr19": 61_431_566,  "chrX": 171_031_299,  "chrY": 91_744_698,
    "chrM": 16_299,
}

# Colours used in the UCSC track hub (R,G,B strings).
MODALITY_COLOURS = {
    "ATAC":             "255,127,14",   # orange
    "DNASE":            "44,160,44",    # green
    "CAGE":             "214,39,40",    # red
    "RNA_SEQ":          "148,103,189",  # purple
    "CHIP_HISTONE":     "31,119,180",   # blue
    "CHIP_TF":          "227,119,194",  # pink
    "SPLICE_SITES":     "188,189,34",   # olive
    "SPLICE_SITE_USAGE":"23,190,207",   # cyan
    "PROCAP":           "140,86,75",    # brown
    "CONTACT_MAPS":     "127,127,127",  # grey
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_prediction_interval(
    target_start: int,
    target_end: int,
    chrom: str,
    input_length: int = MODEL_INPUT_LENGTH,
) -> genome.Interval:
    """Return a *input_length*-bp interval centred on the target region."""
    centre = (target_start + target_end) // 2
    pred_start = centre - input_length // 2
    pred_end = centre + input_length // 2
    return genome.Interval(chrom, pred_start, pred_end)


def sanitize_name(name: str) -> str:
    """Turn an arbitrary track name into a filesystem / trackDb-safe string."""
    return (
        name
        .replace(" ", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("+", "pos")
        .replace("-", "neg")
        .replace(".", "unstr")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace('"', "")
    )


# ═══════════════════════════════════════════════════════════════════════════
# bigWig export
# ═══════════════════════════════════════════════════════════════════════════

def write_bigwig(
    filepath: str | Path,
    chrom: str,
    start: int,
    values: np.ndarray,
    resolution: int,
    chrom_sizes: dict[str, int],
) -> bool:
    """Write a single 1-D signal vector into a bigWig file.

    Parameters
    ----------
    filepath : path to the output .bw file.
    chrom    : chromosome name.
    start    : 0-based genomic start of the first bin.
    values   : 1-D float array (one value per bin).
    resolution : bin width in bp.
    chrom_sizes : {chrom: length} dict.

    Returns True if the file was written, False if skipped (all-zero).
    """
    values = np.asarray(values, dtype=np.float64)

    # Build coordinate arrays.
    n_bins = len(values)
    starts = np.arange(n_bins, dtype=np.int64) * resolution + start
    ends = starts + resolution

    # Keep only valid finite bins that fall within the chromosome.
    chrom_len = chrom_sizes[chrom]
    mask = (starts >= 0) & (ends <= chrom_len) & np.isfinite(values)
    if mask.sum() == 0:
        return False

    bw = pyBigWig.open(str(filepath), "w")
    header = [(c, chrom_sizes[c]) for c in chrom_sizes]
    bw.addHeader(header)

    bw.addEntries(
        [chrom] * int(mask.sum()),
        starts[mask].tolist(),
        ends=ends[mask].tolist(),
        values=values[mask].astype(np.float32).tolist(),
    )
    bw.close()
    return True


def export_trackdata(
    track_data: track_data_module.TrackData | None,
    output_type: dna_output.OutputType,
    mm10_dir: str,
    chrom_sizes: dict[str, int],
) -> list[dict]:
    """Export a TrackData object to one bigWig file per track.

    Returns a list of metadata dicts (one per written file) consumed by
    *create_track_hub()*.
    """
    modality = output_type.name

    if track_data is None:
        print(f"  {modality}: no predictions (skipped)")
        return []

    interval = track_data.interval
    if interval is None:
        print(f"  {modality}: interval missing (skipped)")
        return []

    values = track_data.values
    # Contact maps are 3-D [pos, pos, tracks] – cannot be bigWig.
    if values.ndim > 2:
        print(f"  {modality}: 2-D output – not suitable for bigWig (skipped)")
        return []

    if values.ndim == 1:
        values = values[:, np.newaxis]

    chrom = interval.chromosome
    start = interval.start
    resolution = track_data.resolution
    metadata_df = track_data.metadata

    mod_dir = os.path.join(mm10_dir, modality.lower())
    os.makedirs(mod_dir, exist_ok=True)

    infos: list[dict] = []

    for i in range(track_data.num_tracks):
        track_name = str(metadata_df.iloc[i]["name"])
        track_strand = str(metadata_df.iloc[i]["strand"])
        safe = sanitize_name(f"{modality}_{track_name}_{track_strand}")
        filename = f"{safe}.bw"
        filepath = os.path.join(mod_dir, filename)

        col_values = values[:, i].astype(np.float64)

        # Skip all-zero / negligible tracks.
        if np.nanmax(np.abs(col_values)) < 1e-10:
            continue

        ok = write_bigwig(filepath, chrom, start, col_values, resolution, chrom_sizes)
        if not ok:
            continue

        infos.append(
            {
                "track_id": safe,
                "short_label": f"{track_name} ({track_strand})"[:80],
                "long_label": f"{modality} – {track_name}  strand:{track_strand}"[:255],
                "bigDataUrl": f"{modality.lower()}/{filename}",
                "modality": modality,
                "colour": MODALITY_COLOURS.get(modality, "0,0,0"),
            }
        )

    print(f"  {modality}: {len(infos)} bigWig files written")
    return infos


# ═══════════════════════════════════════════════════════════════════════════
# UCSC Track Hub generation
# ═══════════════════════════════════════════════════════════════════════════

def create_track_hub(
    track_infos: list[dict],
    hub_root: str,
    hub_url: str = "",
) -> None:
    """Write hub.txt, genomes.txt, and mm10/trackDb.txt."""

    os.makedirs(hub_root, exist_ok=True)

    # ── hub.txt ──────────────────────────────────────────────────────────
    with open(os.path.join(hub_root, "hub.txt"), "w") as fh:
        fh.write(
            "hub AlphaGenome_mm10_predictions\n"
            "shortLabel AlphaGenome Predictions\n"
            "longLabel AlphaGenome predicted genomic tracks – "
            "mm10 chr11:32114094-32179698\n"
            "genomesFile genomes.txt\n"
            "email alphagenome@example.com\n"
        )

    # ── genomes.txt ──────────────────────────────────────────────────────
    with open(os.path.join(hub_root, "genomes.txt"), "w") as fh:
        fh.write("genome mm10\ntrackDb mm10/trackDb.txt\n")

    # ── trackDb.txt ──────────────────────────────────────────────────────
    mm10_dir = os.path.join(hub_root, "mm10")
    os.makedirs(mm10_dir, exist_ok=True)

    # Group tracks by modality for composite tracks.
    by_modality: dict[str, list[dict]] = {}
    for info in track_infos:
        by_modality.setdefault(info["modality"], []).append(info)

    lines: list[str] = []
    for modality in sorted(by_modality):
        tracks = by_modality[modality]
        colour = MODALITY_COLOURS.get(modality, "0,0,0")
        comp = f"ag_{modality.lower()}"

        lines.append(f"track {comp}")
        lines.append("compositeTrack on")
        lines.append(f"shortLabel AG {modality}")
        lines.append(f"longLabel AlphaGenome {modality} predictions")
        lines.append("type bigWig")
        lines.append("visibility dense")
        lines.append(f"color {colour}")
        lines.append("autoScale on")
        lines.append("maxHeightPixels 100:32:8")
        lines.append("")

        for t in tracks:
            lines.append(f"    track {t['track_id']}")
            lines.append(f"    parent {comp} off")
            lines.append(f"    shortLabel {t['short_label']}")
            lines.append(f"    longLabel {t['long_label']}")
            lines.append("    type bigWig")
            lines.append(f"    bigDataUrl {t['bigDataUrl']}")
            lines.append(f"    color {t['colour']}")
            lines.append("    autoScale on")
            lines.append("    maxHeightPixels 100:32:8")
            lines.append("    visibility dense")
            lines.append("")

    with open(os.path.join(mm10_dir, "trackDb.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\nTrack Hub written to: {hub_root}/")
    print(f"  hub.txt      -> {hub_root}/hub.txt")
    print(f"  genomes.txt  -> {hub_root}/genomes.txt")
    print(f"  trackDb.txt  -> {mm10_dir}/trackDb.txt")

    if hub_url:
        url = hub_url.rstrip("/")
        print(
            f"\nOpen in UCSC Genome Browser:\n"
            f"  https://genome.ucsc.edu/cgi-bin/hgTracks"
            f"?db=mm10"
            f"&hubUrl={url}/hub.txt"
            f"&position=chr11:32114094-32179698"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaGenome: predict all modalities and create a UCSC Track Hub",
    )
    parser.add_argument(
        "--output-dir",
        default="trackhub_output",
        help="Root directory for the Track Hub (default: trackhub_output)",
    )
    parser.add_argument(
        "--hub-url",
        default="",
        help="Public base URL where you will host the Track Hub "
        "(used to print a clickable UCSC browser link)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu", "tpu"],
        help="JAX device for inference.  Default: auto-detect GPU/TPU, "
        "falls back to CPU with a warning.",
    )
    parser.add_argument(
        "--model-source",
        default="huggingface",
        choices=["huggingface", "kaggle"],
        help="Where to download the model checkpoint from (default: huggingface)",
    )
    parser.add_argument(
        "--model-version",
        default="ALL_FOLDS",
        help="Model version / fold to use (default: ALL_FOLDS). "
        "Options: ALL_FOLDS, FOLD_0, FOLD_1, FOLD_2, FOLD_3",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_version = ag_dna_model.ModelVersion[args.model_version]

    # ── Resolve device ────────────────────────────────────────────────────
    if args.device:
        device = jax.devices(args.device)[0]
    else:
        # Try GPU/TPU first; fall back to CPU with an explicit device handle.
        try:
            device = jax.devices("gpu")[0]
        except RuntimeError:
            try:
                device = jax.devices("tpu")[0]
            except RuntimeError:
                print(
                    "WARNING: No GPU/TPU found.  Running on CPU – "
                    "inference will be slow (~30-60 min)."
                )
                device = jax.devices("cpu")[0]

    print(f"JAX back-ends : {jax.devices()}")
    print(f"Using device  : {device}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading AlphaGenome model ({model_version.name}) "
          f"from {args.model_source} ...")

    try:
        if args.model_source == "kaggle":
            ag_model = ag_dna_model.create_from_kaggle(
                model_version, device=device,
            )
        else:
            ag_model = ag_dna_model.create_from_huggingface(
                model_version, device=device,
            )
    except Exception as exc:
        print(f"\nERROR loading model: {exc}\n")
        print("The AlphaGenome model checkpoint is gated and requires authentication.")
        print("Please configure ONE of the following:\n")
        print("  Option A – HuggingFace:")
        print("    1. Accept the licence at https://huggingface.co/google/alphagenome-all-folds")
        print("    2. export HF_TOKEN=<your-huggingface-token>")
        print("       OR run:  huggingface-cli login\n")
        print("  Option B – Kaggle:")
        print("    1. Accept the licence at https://www.kaggle.com/models/google/alphagenome")
        print("    2. export KAGGLE_USERNAME=<your-username>")
        print("       export KAGGLE_KEY=<your-api-key>")
        print("       OR place ~/.kaggle/kaggle.json\n")
        print("Then re-run this script.")
        sys.exit(1)

    print("Model loaded.\n")

    # ── Determine available outputs for mouse ─────────────────────────────
    metadata = metadata_lib.load(ORGANISM)
    available_outputs: list[dna_output.OutputType] = []
    for ot in ALL_OUTPUT_TYPES:
        md = metadata.get(ot)
        if md is not None:
            padding = metadata.padding.get(ot)
            n_real = int((~padding).sum()) if padding is not None else len(md)
            if n_real > 0:
                available_outputs.append(ot)

    print("Available mouse output types:")
    for ot in available_outputs:
        md = metadata.get(ot)
        padding = metadata.padding.get(ot)
        n_real = int((~padding).sum()) if padding is not None else len(md)
        res = metadata.resolution(ot)
        print(f"  {ot.name:25s}  {n_real:>4d} tracks  @ {res:>4d} bp")

    # ── Prediction interval ───────────────────────────────────────────────
    pred_interval = compute_prediction_interval(
        TARGET_START, TARGET_END, TARGET_CHROM, MODEL_INPUT_LENGTH
    )
    print(f"\nTarget region      : {TARGET_CHROM}:{TARGET_START:,}-{TARGET_END:,}")
    print(f"Prediction window  : {pred_interval}  ({MODEL_INPUT_LENGTH:,} bp)")

    # ── Run predictions ───────────────────────────────────────────────────
    print("\nRunning predictions (this may take a while on CPU) ...")
    output: dna_output.Output = ag_model.predict_interval(
        pred_interval,
        organism=ORGANISM,
        requested_outputs=available_outputs,
        ontology_terms=None,        # predict all available cell-types / tissues
    )
    print("Predictions complete.\n")

    # ── Export bigWig files ────────────────────────────────────────────────
    print("Exporting predictions to bigWig ...")
    mm10_dir = os.path.join(output_dir, "mm10")
    os.makedirs(mm10_dir, exist_ok=True)

    all_track_infos: list[dict] = []
    for ot in BIGWIG_OUTPUT_TYPES:
        td = output.get(ot)
        infos = export_trackdata(td, ot, mm10_dir, MM10_CHROM_SIZES)
        all_track_infos.extend(infos)

    print(f"\nTotal bigWig tracks written: {len(all_track_infos)}")

    # ── Create Track Hub ──────────────────────────────────────────────────
    print("\nCreating UCSC Track Hub ...")
    create_track_hub(all_track_infos, output_dir, hub_url=args.hub_url)

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(
        f"\nTo view in the UCSC Genome Browser:\n"
        f"  1. Host the '{output_dir}/' directory on a public web server\n"
        f"     (e.g. GitHub Pages, an S3 bucket, or any HTTPS-accessible host).\n"
        f"  2. Go to  https://genome.ucsc.edu/cgi-bin/hgTracks?db=mm10\n"
        f"  3. Click 'track hubs' -> 'My Hubs' -> paste your hub.txt URL.\n"
        f"  4. Navigate to chr11:32,114,094-32,179,698.\n"
    )


if __name__ == "__main__":
    main()
