#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Process a single protein pair through Chai-1 WITHOUT crosslink restraints.
Called by the wrapper script for memory isolation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from chai_lab.chai1 import run_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, PROTEOME_CSV, CHAI_MULTIMER_DIR
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_sequence_mapping(df_proteome,
                           id_col="Gene_ID",
                           seq_col="Protein_Sequence") -> dict:
    """
    Build a simple Gene_ID -> sequence lookup from the proteome CSV.
    Warns if any duplicate Gene_IDs are found.
    """
    duplicates = df_proteome[df_proteome.duplicated(subset=id_col, keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {duplicates[id_col].nunique()} duplicate Gene_ID(s) in proteome — "
                       f"keeping first occurrence.")

    mapping = (
        df_proteome.drop_duplicates(subset=id_col, keep="first")
        .set_index(id_col)[seq_col]
        .to_dict()
    )
    logger.info(f"Built sequence mapping for {len(mapping)} proteins")
    return mapping


def check_if_pair_complete(output_dir: Path, protein1: str, protein2: str) -> bool:
    """
    Check if protein pair already has a complete prediction (10 files).
    Checks both A-B and B-A orientations.
    """
    matching_dirs = list(output_dir.glob(f"pair_*_{protein1}_{protein2}"))
    matching_dirs.extend(output_dir.glob(f"pair_*_{protein2}_{protein1}"))

    if not matching_dirs:
        return False

    pair_dir = matching_dirs[0]

    for i in range(5):
        if not (pair_dir / f"pred.model_idx_{i}.cif").exists():
            return False
        if not (pair_dir / f"scores.model_idx_{i}.npz").exists():
            return False

    return True


def process_single_pair(idx, protein1, protein2, base_path):
    """Process a single protein pair through Chai-1"""

    output_dir = CHAI_MULTIMER_DIR

    # Safeguard: skip if already complete
    if (check_if_pair_complete(old_output_dir, protein1, protein2)):
        logger.info(f"Pair ({protein1}-{protein2}) already complete — skipping")
        return

    # Load proteome and build mapping
    proteome_csv = PROTEOME_CSV   # update filename as needed
    df_proteome  = pd.read_csv(proteome_csv)
    seq_mapping  = build_sequence_mapping(df_proteome)

    # Get sequences
    seq1 = seq_mapping.get(protein1)
    seq2 = seq_mapping.get(protein2)

    if not seq1 or not seq2:
        missing = [p for p, s in [(protein1, seq1), (protein2, seq2)] if not s]
        logger.error(f"Missing sequence data for: {missing}")
        sys.exit(1)

    # Setup directories
    chai_fasta_dir = output_dir / "chai_fasta"
    chai_fasta_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create FASTA file
    temp_fasta_path = chai_fasta_dir / f"{idx}_chai_fasta_{protein1}_{protein2}.fasta"

    try:
        with open(temp_fasta_path, 'w') as fasta_file:
            fasta_file.write(f">protein|{protein1}\n{seq1}\n")
            fasta_file.write(f">protein|{protein2}\n{seq2}\n")

        logger.info(f"Created FASTA for {protein1}-{protein2}")

        # Run Chai-1 inference
        pair_output_dir = output_dir / f"pair_{idx}_{protein1}_{protein2}"
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Chai-1 inference for {protein1}-{protein2}")

        chai_run = run_inference(
            fasta_file=temp_fasta_path,
            output_dir=pair_output_dir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            num_diffn_samples=5,
            seed=42,
            device="cuda:0",
            use_esm_embeddings=True,
        )

        logger.info(f"Successfully completed Chai-1 inference for {protein1}-{protein2}")

    except Exception as e:
        logger.error(f"Error processing {protein1}-{protein2}: {e}")
        raise

    finally:
        print("retained FASTA")
        # # Clean up temp FASTA
        # if temp_fasta_path.exists():
        #     try:
        #         temp_fasta_path.unlink()
        #         logger.info(f"Cleaned up temp FASTA: {temp_fasta_path}")
        #     except Exception as cleanup_error:
        #         logger.warning(f"Could not clean up temp FASTA: {cleanup_error}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python 01_naive_multimer_inference.py <idx> <protein1> <protein2>")
        sys.exit(1)

    idx      = int(sys.argv[1])
    protein1 = sys.argv[2]
    protein2 = sys.argv[3]

    base = REPO_ROOT

    logger.info(f"Processing pair {idx}: {protein1}-{protein2}")

    try:
        process_single_pair(idx, protein1, protein2, base)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

