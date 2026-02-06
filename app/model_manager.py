"""Model manager for loading and caching the AlphaGenome model."""

import logging
import threading
from typing import Optional

import numpy as np

from alphagenome.data import genome
from alphagenome.models.dna_output import OutputType
from alphagenome_research.model import dna_model

from app.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AlphaGenome model lifecycle and inference."""

    def __init__(self):
        self._model: Optional[dna_model.AlphaGenomeModel] = None
        self._lock = threading.Lock()
        self._loading = False
        self._loaded = False
        self._error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def error(self) -> Optional[str]:
        return self._error

    def load_model(self) -> None:
        """Load the AlphaGenome model from Hugging Face."""
        with self._lock:
            if self._loaded or self._loading:
                return
            self._loading = True

        try:
            logger.info(
                "Loading AlphaGenome model (version: %s)...",
                settings.model_version,
            )

            if settings.hf_token:
                import huggingface_hub
                huggingface_hub.login(token=settings.hf_token)
                logger.info("Authenticated with Hugging Face.")

            self._model = dna_model.create_from_huggingface(
                settings.model_version
            )
            self._loaded = True
            self._loading = False
            logger.info("AlphaGenome model loaded successfully.")
        except Exception as e:
            self._loading = False
            self._error = str(e)
            logger.error("Failed to load AlphaGenome model: %s", e)
            raise

    def _get_organism(self, organism: str) -> dna_model.Organism:
        """Convert organism string to Organism enum."""
        organism_map = {
            "human": dna_model.Organism.HOMO_SAPIENS,
            "homo_sapiens": dna_model.Organism.HOMO_SAPIENS,
            "mouse": dna_model.Organism.MUS_MUSCULUS,
            "mus_musculus": dna_model.Organism.MUS_MUSCULUS,
        }
        return organism_map.get(organism.lower(), dna_model.Organism.HOMO_SAPIENS)

    def _parse_output_types(self, output_types: list[str]) -> list[OutputType]:
        """Convert output type strings to OutputType enums."""
        type_map = {
            "rna_seq": OutputType.RNA_SEQ,
            "cage": OutputType.CAGE,
            "dnase": OutputType.DNASE,
            "atac": OutputType.ATAC,
            "chip_histone": OutputType.CHIP_HISTONE,
            "chip_tf": OutputType.CHIP_TF,
            "splice_sites": OutputType.SPLICE_SITES,
            "splice_site_usage": OutputType.SPLICE_SITE_USAGE,
            "splice_junctions": OutputType.SPLICE_JUNCTIONS,
            "contact_maps": OutputType.CONTACT_MAPS,
            "procap": OutputType.PROCAP,
        }
        result = []
        for ot in output_types:
            key = ot.lower().strip()
            if key in type_map:
                result.append(type_map[key])
            else:
                raise ValueError(
                    f"Unknown output type: {ot}. "
                    f"Available types: {list(type_map.keys())}"
                )
        return result

    def _track_data_to_dict(self, track_data) -> dict:
        """Convert TrackData object to a JSON-serializable dictionary."""
        if track_data is None:
            return None

        values = np.array(track_data.values)

        # Limit the data for large outputs (downsample if needed)
        max_points = 4096
        if values.ndim >= 2 and values.shape[0] > max_points:
            step = values.shape[0] // max_points
            values = values[::step]

        result = {
            "values": values.tolist(),
            "resolution": int(track_data.resolution),
            "names": list(track_data.names),
            "num_tracks": int(track_data.num_tracks),
        }

        if track_data.interval is not None:
            result["interval"] = {
                "chromosome": track_data.interval.chromosome,
                "start": int(track_data.interval.start),
                "end": int(track_data.interval.end),
            }

        return result

    def _output_to_dict(self, output) -> dict:
        """Convert an Output object to a JSON-serializable dictionary."""
        result = {}

        attr_names = [
            "rna_seq", "cage", "dnase", "atac", "chip_histone",
            "chip_tf", "splice_sites", "splice_site_usage",
            "splice_junctions", "contact_maps", "procap",
        ]

        for attr in attr_names:
            try:
                data = getattr(output, attr, None)
                if data is not None:
                    result[attr] = self._track_data_to_dict(data)
            except Exception:
                pass

        return result

    def predict_interval(
        self,
        chromosome: str,
        start: int,
        end: int,
        organism: str = "human",
        output_types: list[str] = None,
        ontology_terms: list[str] = None,
    ) -> dict:
        """Make a prediction for a genomic interval."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Please wait for model initialization.")

        if output_types is None:
            output_types = ["rna_seq"]

        interval = genome.Interval(
            chromosome=chromosome,
            start=start,
            end=end,
        )

        requested_outputs = self._parse_output_types(output_types)
        org = self._get_organism(organism)

        output = self._model.predict_interval(
            interval=interval,
            organism=org,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

        return self._output_to_dict(output)

    def predict_variant(
        self,
        chromosome: str,
        start: int,
        end: int,
        variant_position: int,
        ref_bases: str,
        alt_bases: str,
        organism: str = "human",
        output_types: list[str] = None,
        ontology_terms: list[str] = None,
    ) -> dict:
        """Make a variant effect prediction."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Please wait for model initialization.")

        if output_types is None:
            output_types = ["rna_seq"]

        interval = genome.Interval(
            chromosome=chromosome,
            start=start,
            end=end,
        )

        variant = genome.Variant(
            chromosome=chromosome,
            position=variant_position,
            reference_bases=ref_bases,
            alternate_bases=alt_bases,
        )

        requested_outputs = self._parse_output_types(output_types)
        org = self._get_organism(organism)

        variant_output = self._model.predict_variant(
            interval=interval,
            variant=variant,
            organism=org,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

        return {
            "reference": self._output_to_dict(variant_output.reference),
            "alternate": self._output_to_dict(variant_output.alternate),
        }

    def predict_sequence(
        self,
        sequence: str,
        organism: str = "human",
        output_types: list[str] = None,
        ontology_terms: list[str] = None,
    ) -> dict:
        """Make a prediction from a raw DNA sequence."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Please wait for model initialization.")

        if output_types is None:
            output_types = ["rna_seq"]

        requested_outputs = self._parse_output_types(output_types)
        org = self._get_organism(organism)

        output = self._model.predict_sequence(
            sequence=sequence,
            organism=org,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

        return self._output_to_dict(output)


# Global singleton
model_manager = ModelManager()
