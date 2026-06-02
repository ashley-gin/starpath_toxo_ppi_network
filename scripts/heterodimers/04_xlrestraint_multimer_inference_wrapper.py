#!/usr/bin/env python
# coding: utf-8

"""
Wrapper script to run Chai-1 inference WITH crosslink restraints.
Iterates over all crosslinks (one at a time per PPI) and identifies each
by its crosslink residue identity and position (e.g. K329_Y45)
Each pair runs in a separate subprocess for memory isolation.
"""

import subprocess
import sys
import logging
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, PROTEOME_CSV, PPI_CSV, CHAI_MULTIMER_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """
    Load proteome and XL-MS data.
    NOTE: Uses the XL-MS data after sorting by subsets for the "inter_xl_csv" entry.
    Checks ensure this is only interprotein XL-MS data (no intraprotein).
    """
    try:     
        proteome_csv = PROTEOME_CSV
        df_rh88_proteome = pd.read_csv(proteome_csv)

        inter_xl_csv = PROCESSED_DATA_DIR / "by_crosslink" / "step1_unconfident_multimer_ranked_by_monomer_pae_one_confident.csv"
        inter_xl_pairs = pd.read_csv(inter_xl_csv)

        logger.info(f"Loaded {len(df_rh88_proteome)} proteins from proteome")
        logger.info(f"Loaded {len(inter_xl_pairs)} crosslink entries from XL-MS data")

        return df_rh88_proteome, inter_xl_pairs
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def prepare_crosslinks(inter_xl_data):
    """
    Prepare all crosslinks for iteration. No ranking is applied — crosslinks are
    iterated in the order they appear in the input data. Each crosslink is
    uniquely identified by its residue identity and position on each chain
    (e.g. K329_Y45)

    Returns a DataFrame with a stable 'xl_id' column added.
    """
    df = inter_xl_data.copy()
    df.columns = df.columns.str.strip()
    
    required_cols = ['Protein A', 'Protein B', 'Crosslink Position A', 'Crosslink Position B',
                     'Crosslinked Residue A', 'Crosslinked Residue B']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                         f"Available columns: {list(df.columns)}")
    
    # Build stable crosslink ID from residue identity + position
    df['xl_id'] = (
        df['Crosslinked Residue A'].astype(str).str.strip()
        + df['Crosslink Position A'].apply(lambda x: str(int(float(x))))
        + "_"
        + df['Crosslinked Residue B'].astype(str).str.strip()
        + df['Crosslink Position B'].apply(lambda x: str(int(float(x))))
    )

    # Normalized PPI key for logging/summary (order-independent)
    df['_interaction_key'] = df.apply(
        lambda row: "|".join(sorted([str(row['Protein A']).strip(),
                                     str(row['Protein B']).strip()])),
        axis=1
    )

    unique_ppis = df['_interaction_key'].nunique()
    logger.info(f"Total crosslink entries: {len(df)}")
    logger.info(f"Unique PPIs: {unique_ppis}")
    logger.info(f"Average crosslinks per PPI: {len(df) / unique_ppis:.2f}")
    
    return df

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

    for i in range(5):
        for filename in [f"pred.model_idx_{i}.cif", f"scores.model_idx_{i}.npz"]:
            if not (pair_dir / filename).exists():
                return False

    return True

def main():
    
    # Load data
    df_rh88_proteome, inter_xl_pairs = load_data()
    
    # Prepare crosslinks (no ranking, just stable xl_id assignment)
    all_crosslinks = prepare_crosslinks(inter_xl_pairs)
    
    # Create output directory
    final_output_dir = CHAI_MULTIMER_DIR / 'single_interXL_chai_summary'
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    chai_output_dir = CHAI_MULTIMER_DIR / "xlrestraint_chai_outputs"
    chai_output_dir.mkdir(parents=True, exist_ok=True)

    # Find incomplete crosslinks
    incomplete_crosslinks = []
    for idx, row in all_crosslinks.iterrows():
        protein1 = str(row['Protein A']).strip()
        protein2 = str(row['Protein B']).strip()
        xl_id = row['xl_id']
        
        if not check_if_pair_complete(chai_output_dir, protein1, protein2, xl_id):
            residue_a = str(row['Crosslinked Residue A']).strip()
            pos_a = int(float(row['Crosslink Position A']))
            residue_b = str(row['Crosslinked Residue B']).strip()
            pos_b = int(float(row['Crosslink Position B']))
            
            incomplete_crosslinks.append((
                idx, protein1, protein2,
                residue_a, pos_a, residue_b, pos_b,
                xl_id
            ))
    
    total_crosslinks = len(all_crosslinks)
    complete = total_crosslinks - len(incomplete_crosslinks)
    unique_ppis = all_crosslinks['_interaction_key'].nunique()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total unique PPIs: {unique_ppis}")
    logger.info(f"Total crosslinks to process: {total_crosslinks}")
    logger.info(f"Already complete: {complete}")
    logger.info(f"Remaining to process: {len(incomplete_crosslinks)}")
    logger.info(f"{'='*60}\n")
    
    # Process each incomplete crosslink in a separate subprocess
    failed_crosslinks = []
    
    for i, crosslink_data in enumerate(incomplete_crosslinks, 1):
        (idx, protein1, protein2,
         residue_a, pos_a, residue_b, pos_b,
         xl_id) = crosslink_data
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing crosslink {i}/{len(incomplete_crosslinks)}")
        logger.info(f"PPI: {protein1}-{protein2}")
        logger.info(f"Crosslink ID: {xl_id}")
        logger.info(f"Crosslink: {residue_a}{pos_a} <-> {residue_b}{pos_b}")
        # logger.info(f"Distance range: {dist_min}-{dist_max} Å (mean: {dist_mean} Å)")
        logger.info(f"{'='*60}")
        
        try:
            result = subprocess.run(
                [
                    sys.executable, 
                    "05_xlrestraint_multimer_inference.py",
                    str(idx), protein1, protein2,
                    residue_a, str(pos_a), residue_b, str(pos_b)
                    # Note: crosslink_rank removed; xl_id is derived inside single script
                ],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per pair
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully processed {protein1}-{protein2} (XL: {xl_id})")
            else:
                logger.error(f"Failed to process {protein1}-{protein2} (XL: {xl_id})")
                logger.error(f"STDERR: {result.stderr}")
                failed_crosslinks.append((idx, protein1, protein2, xl_id, result.stderr))
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {protein1}-{protein2} (XL: {xl_id})")
            failed_crosslinks.append((idx, protein1, protein2, xl_id, "Timeout after 1 hour"))
        except Exception as e:
            logger.error(f"Error processing {protein1}-{protein2} (XL: {xl_id}): {e}")
            failed_crosslinks.append((idx, protein1, protein2, xl_id, str(e)))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL PROCESSING SUMMARY")
    print("="*60)
    print(f"Total unique PPIs in dataset: {unique_ppis}")
    print(f"Total crosslinks in dataset: {total_crosslinks}")
    print(f"Already complete at start: {complete}")
    print(f"Attempted to process: {len(incomplete_crosslinks)}")
    print(f"Successfully processed: {len(incomplete_crosslinks) - len(failed_crosslinks)}")
    print(f"Failed: {len(failed_crosslinks)}")
    print(f"Total now complete: {complete + len(incomplete_crosslinks) - len(failed_crosslinks)}")
    print("="*60)
    
    if failed_crosslinks:
        print("\nFailed crosslinks:")
        for idx, protein1, protein2, xl_id, error in failed_crosslinks:
            print(f"  {idx}: {protein1}-{protein2} (XL: {xl_id})")
            print(f"     Error: {error[:100]}...")
    
    # Save summary to file
    summary_file = final_output_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CHAI-1 PROCESSING SUMMARY WITH INDIVIDUAL CROSSLINK RESTRAINTS\n")
        f.write("="*60 + "\n")
        f.write(f"Total unique PPIs: {unique_ppis}\n")
        f.write(f"Total crosslinks: {total_crosslinks}\n")
        f.write(f"Successfully processed: {complete + len(incomplete_crosslinks) - len(failed_crosslinks)}\n")
        f.write(f"Failed: {len(failed_crosslinks)}\n\n")
        
        if failed_crosslinks:
            f.write("Failed crosslinks:\n")
            for idx, protein1, protein2, xl_id, error in failed_crosslinks:
                f.write(f"{idx}: {protein1}-{protein2} XL {xl_id}\n")
                f.write(f"  Error: {error}\n\n")
    
    logger.info(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()

