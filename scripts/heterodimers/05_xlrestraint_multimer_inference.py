#!/usr/bin/env python
# coding: utf-8

"""
Process a single protein pair through Chai-1 WITH crosslink restraints.
Called by the wrapper script for memory isolation.
Pairs are identified by crosslink residue identity and position (e.g. K329_Y45)
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

def make_xl_id(residue_a, pos_a, residue_b, pos_b):
    """
    Create a stable crosslink identifier string from residue identity and position.
    e.g. 'K329_Y45'
    """
    return f"{residue_a}{pos_a}_{residue_b}{pos_b}"

def check_if_pair_complete(output_dir: Path, protein1: str, protein2: str, xl_id: str) -> bool:
    """
    Check if protein pair already has a complete prediction (10 output files).
    Matches on crosslink identity (e.g. K329_Y45) rather than distance rank.
    """
    matching_dirs = list(output_dir.glob(f"pair_*_{protein1}_{protein2}_{xl_id}"))
    matching_dirs.extend(output_dir.glob(f"pair_*_{protein2}_{protein1}_{xl_id}"))

    if not matching_dirs:
        return False

    pair_dir = matching_dirs[0]

    # Expect 10 files: pred + scores for each of 5 models
    for i in range(5):
        for filename in [f"pred.model_idx_{i}.cif", f"scores.model_idx_{i}.npz"]:
            if not (pair_dir / filename).exists():
                return False

    return True

def create_restraint_file(restraint_path, residue_a, pos_a, residue_b, pos_b):
    """
    Create a CSV restraint file for Chai-1.
    
    Args:
        restraint_path: Path to save the restraint CSV file
        residue_a: Single letter amino acid code for chain A (e.g., 'K')
        pos_a: Position in chain A (int)
        residue_b: Single letter amino acid code for chain B (e.g., 'Y')
        pos_b: Position in chain B (int)
        dist_min: 0
        dist_max: 20
    """
    # Accommodate "N-term" crosslinking residue position
    if residue_a == "N-term": 
        residue_a = "M"    
    if residue_b == "N-term": 
        residue_b = "M"

    restraint_data = {
        'chainA': ['A'],
        'res_idxA': [f'{residue_a}{pos_a}'],
        'chainB': ['B'],
        'res_idxB': [f'{residue_b}{pos_b}'],
        'connection_type': ['contact'],
        'confidence': [1.0],
        'min_distance_angstrom': [0.0],
        'max_distance_angstrom': [20.0], #modified to max 30 cutoff due to Chai-1 training being Gaussian of 6-30 angstrom
        'comment': ['single XL restraint'],
        'restraint_id': ['restraint_1']
    }
    
    df_restraint = pd.DataFrame(restraint_data)
    df_restraint.to_csv(restraint_path, index=False)
    logger.info(f"Created restraint file: {restraint_path}")
    logger.info(f"Restraint: A-{residue_a}{pos_a} <-> B-{residue_b}{pos_b}")

def process_single_pair(idx, protein1, protein2, base_path, 
                        residue_a, pos_a, residue_b, pos_b):
    """Process a single protein pair through Chai-1 with restraints"""

    xl_id = make_xl_id(residue_a, pos_a, residue_b, pos_b)

    # Check if already complete
    new_output_dir = CHAI_MULTIMER_DIR / 'xlrestraint_chai_outputs'
    if check_if_pair_complete(new_output_dir, protein1, protein2, xl_id):
        logger.info(f"Pair ({protein1}-{protein2}) crosslink {xl_id} already complete - skipping")
        return
    
    # Load proteome
    rh88_proteome_csv = PROTEOME_CSV
    df_rh88_proteome = pd.read_csv(rh88_proteome_csv)
    
    # Create mapping
    mapping_by_gene = build_sequence_mapping(df_rh88_proteome)
    
    # Get sequences
    seq1 = mapping_by_gene.get(protein1)
    seq2 = mapping_by_gene.get(protein2)
    
    if not seq1 or not seq2:
        logger.error(f"Missing sequence data for {protein1} or {protein2}")
        sys.exit(1)
    
    # Setup directories
    chai_fasta_dir = CHAI_MULTIMER_DIR / 'xlrestraint_chai_outputs' / 'chai_interXL_fasta'
    chai_fasta_dir.mkdir(parents=True, exist_ok=True)
    
    chai_output_dir = CHAI_MULTIMER_DIR / 'xlrestraint_chai_outputs'
    chai_output_dir.mkdir(parents=True, exist_ok=True)
    
    chai_restraints_dir = CHAI_MULTIMER_DIR / 'xlrestraint_chai_outputs' / 'chai_interXL_restraints'
    chai_restraints_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FASTA and restraint filenames using xl_id
    temp_fasta_path = chai_fasta_dir / f"{idx}_chai_fasta_{protein1}_{protein2}_{xl_id}.fasta"
    temp_restraint_path = chai_restraints_dir / f"{idx}_chai_restraint_{protein1}_{protein2}_{xl_id}.csv"
    
    try:
        # Write FASTA
        with open(temp_fasta_path, 'w') as fasta_file:
            fasta_file.write(f">protein|{protein1}\n{seq1}\n")
            fasta_file.write(f">protein|{protein2}\n{seq2}\n")
        logger.info(f"Created FASTA for {protein1}-{protein2} (XL: {xl_id})")
        
        # Write restraint file
        create_restraint_file(
            temp_restraint_path,
            residue_a, pos_a,
            residue_b, pos_b
        )
        
        # Run Chai-1 inference with restraints
        pair_output_dir = chai_output_dir / f"pair_{idx}_{protein1}_{protein2}_{xl_id}"
        pair_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Chai-1 inference for {protein1}-{protein2} (XL: {xl_id}) with restraints")
        
        run_inference(
            fasta_file=temp_fasta_path,
            output_dir=pair_output_dir,
            constraint_path=temp_restraint_path,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            num_diffn_samples=5,
            seed=42, 
            device="cuda:0",
            use_esm_embeddings=True,
        )
        
        logger.info(f"Successfully completed Chai-1 inference for {protein1}-{protein2} (XL: {xl_id})")
        
    except Exception as e:
        logger.error(f"Error processing {protein1}-{protein2} (XL: {xl_id}): {e}")
        raise
    
    finally:
        logger.info(f"Saved FASTA and restraint files for debugging")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python 05_xlrestraint_multimer_inference.py <idx> <protein1> <protein2> "
              "<residue_a> <pos_a> <residue_b> <pos_b>")
        sys.exit(1)
    
    idx = int(sys.argv[1])
    protein1 = sys.argv[2]
    protein2 = sys.argv[3]
    residue_a = sys.argv[4]
    pos_a = int(sys.argv[5])
    residue_b = sys.argv[6]
    pos_b = int(sys.argv[7])
    # dist_min = float(sys.argv[8])
    # dist_max = float(sys.argv[9])
    # Note: crosslink_rank argument removed; identity is now residue+position

    xl_id = make_xl_id(residue_a, pos_a, residue_b, pos_b)
    
    base = REPO_ROOT
    
    logger.info(f"Processing pair {idx}: {protein1}-{protein2} (XL: {xl_id})")
    logger.info(f"Restraint: {residue_a}{pos_a} <-> {residue_b}{pos_b}")
    
    try:
        process_single_pair(idx, protein1, protein2, base,
                            residue_a, pos_a, residue_b, pos_b)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

