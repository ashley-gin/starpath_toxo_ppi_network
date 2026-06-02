#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
Wrapper script to run Chai-1 inference one monomer at a time
NOTE: this is running individual monomers of ENTIRE PPI dataset (regardless of empirical intra or inter XLs)
Each monomer runs in a separate subprocess for complete memory isolation.
"""

import subprocess
import sys
import logging
from pathlib import Path
import pandas as pd
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, PROTEOME_CSV, PPI_CSV, CHAI_MONOMER_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load proteome and XL-MS data"""
    try:     
        rh88_proteome_csv = PROTEOME_CSV
        df_rh88_proteome = pd.read_csv(rh88_proteome_csv)
        
        all_ppi_csv = PPI_CSV
        all_ppi_data = pd.read_csv(all_ppi_csv)

        logger.info(f"Loaded {len(df_rh88_proteome)} proteins from proteome")
        logger.info(f"Loaded {len(all_ppi_data)} protein pairs from XL-MS data")

        return df_rh88_proteome, all_ppi_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def get_unique_proteins(all_ppi_data):
    """Get unique proteins from intraprotein XLs
        separates protein A and protein B
        creates a unique list of all proteins
    """
    #concatenate all proteins from both columns Protein A and Protein B
    all_proteins = pd.concat([
        all_ppi_data['Protein A'],
        all_ppi_data['Protein B']
    ])

    #get unique proteins from concatenated list to prevent duplicate predictions
    unique_proteins = all_proteins.drop_duplicates().sort_values().reset_index(drop=True)
    
    logger.info(f"Found {len(unique_proteins)} unique proteins from {len(all_proteins)} total proteins in {len(all_ppi_data)} PPI entries")
    return unique_proteins

def check_if_monomer_complete(output_dir: Path, protein: str, idx: int) -> bool:
    """Check if monomer already has complete predictions"""
    monomer_dir = output_dir / f"monomer_{idx}_{protein}"
    
    if not monomer_dir.exists():
        return False
    
    # Check for all 10 files (models 0-4)
    for i in range(5):
        cif_file = monomer_dir / f"pred.model_idx_{i}.cif"
        npz_file = monomer_dir / f"scores.model_idx_{i}.npz"
        if not cif_file.exists() or not npz_file.exists():
            return False
            
    return True

def main():
    #parse argument for testing smaller subset of proteins
    parser = argparse.ArgumentParser(description='Run Chai-1 inference on unique protein monomers')
    parser.add_argument('--test', type=int, metavar='N', 
                        help='Test mode: only process first N proteins')
    args = parser.parse_args()
 
    # Load data
    df_rh88_proteome, all_ppi_data = load_data()
    unique_proteins = get_unique_proteins(all_ppi_data)
    
    #Apply test mode if specified
    if args.test:
        logger.info(f"TEST MODE: Limiting to first {args.test} proteins")
        unique_proteins = unique_proteins.head(args.test)

    # Find incomplete monomers
    incomplete_proteins = []
    for idx, protein in enumerate(unique_proteins):
        if not check_if_monomer_complete(CHAI_MONOMER_DIR, protein, idx):
            incomplete_proteins.append((idx, protein))
    
    total = len(unique_proteins)
    complete = total - len(incomplete_proteins)
    
    logger.info(f"Total unique proteins: {total}")
    logger.info(f"Already complete: {complete}")
    logger.info(f"Remaining to process: {len(incomplete_proteins)}")
    
    # Process each incomplete monomer in a separate subprocess
    failed_proteins = []
    
    for i, (idx, protein) in enumerate(incomplete_proteins, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing monomer {i}/{len(incomplete_proteins)}: {protein} (idx={idx})")
        logger.info(f"{'='*60}")
        
        try:
            # Run the single monomer processor script
            result = subprocess.run(
                [sys.executable, "01_naive_monomer_inference.py", str(idx), protein],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per monomer
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully processed {protein}")
            else:
                logger.error(f"Failed to process {protein}")
                logger.error(f"STDERR: {result.stderr}")
                failed_proteins.append((idx, protein, result.stderr))
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {protein}")
            failed_proteins.append((idx, protein, "Timeout after 1 hour"))
        except Exception as e:
            logger.error(f"Error processing {protein}: {e}")
            failed_proteins.append((idx, protein, str(e)))
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL PROCESSING SUMMARY")
    print("="*60)
    print(f"Total unique proteins in dataset: {total}")
    print(f"Already complete at start: {complete}")
    print(f"Attempted to process: {len(incomplete_proteins)}")
    print(f"Successfully processed: {len(incomplete_proteins) - len(failed_proteins)}")
    print(f"Failed: {len(failed_proteins)}")
    print(f"Total now complete: {complete + len(incomplete_proteins) - len(failed_proteins)}")
    print("="*60)
    
    if failed_proteins:
        print("\nFailed proteins:")
        for idx, protein, error in failed_proteins:
            print(f"  {idx}: {protein}")
            print(f"     Error: {error[:100]}...")

        # Save failed proteins to CSV
        failed_df = pd.DataFrame(failed_proteins, columns=['idx', 'protein', 'error'])
        failed_csv_path = PROCESSED_DATA_DIR / "failed_proteins_log.csv"
        failed_df.to_csv(failed_csv_path, index=False)
        logger.info(f"\n Saved failed proteins to: {failed_csv_path}")
        
        # Also save a simple text file with just protein names for easy reprocessing
        failed_txt_path = PROCESSED_DATA_DIR / "failed_proteins_list.txt"
        with open(failed_txt_path, 'w') as f:
            for idx, protein, _ in failed_proteins:
                f.write(f"{protein}\n")
        logger.info(f"✓ Saved failed protein list to: {failed_txt_path}")

if __name__ == "__main__":
    main()

