#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Process a single protein homodimer through Chai-1.
Called by the wrapper script for memory isolation.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from chai_lab.chai1 import run_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, PROTEOME_CSV, CHAI_HOMODIMER_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mapping(df_proteome):
    """Map Gene_ID (e.g. TGRH88_*) to Protein_Sequence (ToxoDB_TGRH88_Protein_Sequences.csv)."""
    mapping_by_gene = {}

    for _, row in df_proteome.iterrows():
        gene_id = row["Gene_ID"]
        sequence = row["Protein_Sequence"]

        if pd.isna(gene_id):
            continue
        gene_id = str(gene_id).strip()
        if not gene_id:
            continue

        if pd.isna(sequence) or len(str(sequence).strip()) == 0:
            logger.warning(f"Empty sequence for gene: {gene_id}")
            continue

        sequence = str(sequence).strip()

        if gene_id in mapping_by_gene:
            if mapping_by_gene[gene_id]["sequence"] != sequence:
                logger.warning(
                    f"Conflict: {gene_id} appears with different sequences; keeping first"
                )
            continue

        mapping_by_gene[gene_id] = {
            "uniprot_id": gene_id,
            "sequence": sequence,
            "all_gene_names": [gene_id],
        }

    return mapping_by_gene

def check_if_homodimer_complete(output_dir: Path, protein: str, idx: int) -> bool:
    """Check if protein homodimer already has complete predictions"""
    homodimer_dir = output_dir / f"homodimer_{idx}_{protein}"
    
    if not homodimer_dir.exists():
        return False
    
    # Check for all 10 files (models 0-4)
    for i in range(5):
        cif_file = homodimer_dir / f"pred.model_idx_{i}.cif"
        npz_file = homodimer_dir / f"scores.model_idx_{i}.npz"
        if not cif_file.exists() or not npz_file.exists():
            return False
    
    return True

def process_single_homodimer(idx, protein, base_path):
    """Process a single protein homodimer through Chai-1"""

    # Check if already complete
    chai_output_dir = CHAI_HOMODIMER_DIR
    if check_if_homodimer_complete(chai_output_dir, protein, idx):
        logger.info(f"Homodimer {idx} ({protein}) already complete - skipping")
        return  # Exit early without processing
    
    # Load proteome
    rh88_proteome_csv = PROTEOME_CSV
    df_rh88_proteome = pd.read_csv(rh88_proteome_csv)
    
    # Create mapping
    mapping_by_gene = mapping(df_rh88_proteome)
    
    # Get sequences
    seq = mapping_by_gene.get(protein)
    
    if not seq:
        logger.error(f"Missing sequence data for {protein}")
        sys.exit(1)
    
    # Setup directories
    chai_fasta_dir = Path(CHAI_HOMODIMER_DIR / 'chai_fasta_homodimers')
    chai_fasta_dir.mkdir(parents=True, exist_ok=True)
    
    chai_output_dir = CHAI_HOMODIMER_DIR
    chai_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FASTA file
    temp_fasta_filename = f"{idx}_chai_fasta_{protein}.fasta"
    temp_fasta_path = chai_fasta_dir / temp_fasta_filename
    
    try:
        with open(temp_fasta_path, 'w') as fasta_file:
            fasta_file.write(f">protein|{protein}_a\n{seq['sequence']}\n")
            fasta_file.write(f">protein|{protein}_b\n{seq['sequence']}\n")
            
        logger.info(f"Created FASTA for {protein}")
        
        # Run Chai-1 inference
        homodimer_output_dir = chai_output_dir / f"homodimer_{idx}_{protein}"
        homodimer_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Chai-1 inference for {protein}")
        
        chai_run = run_inference(
            fasta_file=temp_fasta_path,
            output_dir=homodimer_output_dir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            num_diffn_samples=5,
            seed=42, 
            device="cuda:0",
            use_esm_embeddings=True,
        )
        
        logger.info(f"Successfully completed Chai-1 inference for {protein}")
        
    except Exception as e:
        logger.error(f"Error processing {protein}: {e}")
        raise
    
    finally:
        # Clean up temp FASTA
        if temp_fasta_path.exists():
            try:
                temp_fasta_path.unlink()
                logger.info(f"Cleaned up temp file: {temp_fasta_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temp file: {cleanup_error}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python single_inference_homodimers_chai_20260424.py <idx> <protein>")
        sys.exit(1)
    
    idx = int(sys.argv[1])
    protein = sys.argv[2]
    
    base = REPO_ROOT
    
    logger.info(f"Processing homodimer {idx}: {protein}") 
    
    try:
        process_single_homodimer(idx, protein, base)
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)  # Failure

