# AlphaGenome Predictions – mm10 chr11:32,114,094–32,179,698

This repository contains a script to run **AlphaGenome** (Google DeepMind's
multi-modal genomics model) on a mouse genome region and produce a **UCSC Track
Hub** of all predicted genomic tracks.

## Target Region

- **Genome**: mm10 (GRCm38, *Mus musculus*)
- **Coordinates**: `chr11:32,114,094-32,179,698` (65,604 bp)
- **Prediction window**: 131,072 bp (2^17) centred on the target region

## How AlphaGenome Works

AlphaGenome is a deep-learning model that takes a DNA sequence as input and
simultaneously predicts multiple functional genomics modalities:

```
DNA sequence (131,072 bp, one-hot encoded)
        │
        ▼
┌───────────────────┐
│  Sequence Encoder  │  Conv blocks: 1bp → 128bp
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Transformer Tower  │  9 layers with pairwise attention
└───────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────────┐  ┌──────────────────────┐
│ Sequence Decoder  │  │  128bp / Pair Heads   │
│ (back to 1bp)     │  │  (ChIP, Contact Maps) │
└───────────────────┘  └──────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│ 1bp-resolution Heads             │
│ (ATAC, DNase, CAGE, RNA-seq,     │
│  PRO-cap, Splice sites, Usage)   │
└──────────────────────────────────┘
```

### Predicted Modalities (for mouse)

| Modality          | # Tracks | Resolution | Description                          |
|-------------------|----------|------------|--------------------------------------|
| ATAC-seq          | 18       | 1 bp       | Chromatin accessibility               |
| DNase-seq         | 67       | 1 bp       | DNase I hypersensitivity              |
| CAGE              | 188      | 1 bp       | Gene expression (stranded)            |
| RNA-seq           | 173      | 1 bp       | Gene expression (stranded)            |
| ChIP-seq histone  | 183      | 128 bp     | Histone modifications                 |
| ChIP-seq TF       | 127      | 128 bp     | Transcription factor binding          |
| Splice sites      | 4        | 1 bp       | Donor/acceptor classification         |
| Splice-site usage | 180      | 1 bp       | Fraction of splice-site utilisation   |
| Contact maps      | 8        | 2048 bp    | 3D chromatin contacts (2D, not bigWig)|
| PRO-cap           | 0        | 1 bp       | Not available for mouse               |

Each track corresponds to a specific cell type / tissue / assay combination
identified by an ontology term (e.g., `CL:0000553`).

### Key API Entry Points

```python
from alphagenome_research.model import dna_model as ag_dna_model
from alphagenome.models import dna_model, dna_output
from alphagenome.data import genome

# Load model (requires authentication – see below)
model = ag_dna_model.create_from_huggingface(
    ag_dna_model.ModelVersion.ALL_FOLDS,
    device=jax.devices("gpu")[0],  # or "cpu" for CPU inference
)

# Predict for an interval
interval = genome.Interval("chr11", 32_081_360, 32_212_432)
output = model.predict_interval(
    interval,
    organism=dna_model.Organism.MUS_MUSCULUS,
    requested_outputs=[dna_output.OutputType.ATAC, dna_output.OutputType.DNASE, ...],
    ontology_terms=None,  # all cell types
)

# Access predictions
atac_data = output.atac          # TrackData with .values, .metadata, .interval
dnase_data = output.dnase        # ...same structure
print(atac_data.values.shape)    # (131072, 18) – [positions, tracks]
print(atac_data.metadata)        # DataFrame with 'name', 'strand', etc.
```

## Setup

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install AlphaGenome Research
pip install git+https://github.com/google-deepmind/alphagenome_research.git

# Install bigWig support
pip install pyBigWig
```

### 2. Authenticate for model download

The AlphaGenome checkpoint is a gated model. You need ONE of:

**Option A – HuggingFace** (recommended):
1. Accept the licence at https://huggingface.co/google/alphagenome-all-folds
2. Set your token:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   # OR: huggingface-cli login
   ```

**Option B – Kaggle**:
1. Accept the licence at https://www.kaggle.com/models/google/alphagenome
2. Set your credentials:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

### 3. Run the prediction script

```bash
# GPU (recommended, ~2 minutes inference)
python predict_and_trackhub.py

# CPU (slow, ~30-60 minutes)
python predict_and_trackhub.py --device cpu

# Specify where the hub will be hosted (for direct UCSC browser link)
python predict_and_trackhub.py --hub-url https://your-server.com/trackhub_output
```

### 4. View in UCSC Genome Browser

1. Host the `trackhub_output/` directory on a public HTTPS server
   (e.g., GitHub Pages, AWS S3, Google Cloud Storage, nginx).
2. Go to https://genome.ucsc.edu/cgi-bin/hgTracks?db=mm10
3. Click **My Data** → **Track Hubs** → **My Hubs**
4. Paste the URL to your `hub.txt` (e.g., `https://your-server.com/trackhub_output/hub.txt`)
5. Navigate to `chr11:32,114,094-32,179,698`

## Output Structure

```
trackhub_output/
├── hub.txt                     # Track Hub descriptor
├── genomes.txt                 # Genome assembly (mm10)
└── mm10/
    ├── trackDb.txt             # Track definitions (composite tracks per modality)
    ├── atac/                   # ATAC-seq bigWig files
    │   ├── ATAC_CL_0000553_ATACnegseq_unstr.bw
    │   └── ...
    ├── dnase/                  # DNase-seq bigWig files
    ├── cage/                   # CAGE bigWig files
    ├── rna_seq/                # RNA-seq bigWig files
    ├── chip_histone/           # Histone ChIP-seq bigWig files
    ├── chip_tf/                # TF ChIP-seq bigWig files
    ├── splice_sites/           # Splice site classification bigWig files
    └── splice_site_usage/      # Splice site usage bigWig files
```

## Command-Line Options

| Flag               | Default       | Description                                |
|--------------------|---------------|--------------------------------------------|
| `--output-dir`     | `trackhub_output` | Root directory for track hub files     |
| `--hub-url`        | (none)        | Public URL for the track hub               |
| `--device`         | auto-detect   | `cpu`, `gpu`, or `tpu`                     |
| `--model-source`   | `huggingface` | `huggingface` or `kaggle`                  |
| `--model-version`  | `ALL_FOLDS`   | `ALL_FOLDS`, `FOLD_0`..`FOLD_3`           |
