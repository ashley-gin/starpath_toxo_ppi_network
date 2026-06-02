#!/usr/bin/env python
# coding: utf-8

"""
Wrapper script to run Chai-1 inference WITHOUT crosslink restraints.
Each PPI pair runs in a separate subprocess for complete memory isolation.
(input .csv is pre-filtered to include unique PPI pairs only)
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
    """Load proteome and PPI data"""
    try:
        proteome_csv = PROTEOME_CSV  # update filename as needed
        df_proteome = pd.read_csv(proteome_csv)

        ppi_incomplete_csv = PPI_CSV
        ppi_incomplete = pd.read_csv(ppi_incomplete_csv, index_col = "original_ppi_index")

        logger.info(f"Loaded {len(df_proteome)} proteins from proteome")
        logger.info(f"Loaded {len(ppi_incomplete)} incomplete PPI pairs to process")

        return df_proteome, ppi_incomplete

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

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


def classify_failure(error_text: str) -> str:
    """
    Classify a failure reason from stderr/exception text.
    Returns a short human-readable category string.
    """
    error_lower = error_text.lower()

    if any(kw in error_lower for kw in ["token", "sequence length", "too long",
                                         "exceeds maximum", "max_seq", "tokenlen"]):
        return "token_limit_exceeded"
    elif any(kw in error_lower for kw in ["out of memory", "cuda out of memory",
                                           "oom", "memory error"]):
        return "gpu_out_of_memory"
    elif "timeout" in error_lower:
        return "timeout"
    elif any(kw in error_lower for kw in ["missing sequence", "keyerror", "not found in mapping"]):
        return "missing_sequence"
    elif any(kw in error_lower for kw in ["cuda", "device", "gpu"]):
        return "gpu_error"
    elif any(kw in error_lower for kw in ["connection", "network", "http"]):
        return "network_error"
    else:
        return "unknown_error"


def main():

    old_output_dir = CHAI_MULTIMER_DIR

    # Load data
    df_proteome, ppi_incomplete = load_data()

    # Find pairs that are still incomplete across both directories
    # (kept as a safeguard in case any were completed between runs)
    pairs_to_run = []
    already_complete = 0

    for _, row in ppi_incomplete.iterrows():
        idx      = row.name          # use original PPI df index
        protein1 = row['Protein A']
        protein2 = row['Protein B']

        if (check_if_pair_complete(old_output_dir, protein1, protein2)):
            already_complete += 1
        else:
            pairs_to_run.append((idx, protein1, protein2))

    logger.info(f"Incomplete pairs loaded    : {len(ppi_incomplete)}")
    logger.info(f"Already complete (skipping): {already_complete}")
    logger.info(f"Remaining to process       : {len(pairs_to_run)}")

    # Process each pair in a separate subprocess
    # Each entry: (idx, protein1, protein2, failure_reason, full_error)
    failed_pairs = []

    for i, (idx, protein1, protein2) in enumerate(pairs_to_run, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing pair {i}/{len(pairs_to_run)}: {protein1}-{protein2} (idx={idx})")
        logger.info(f"{'='*60}")

        try:
            result = subprocess.run(
                [sys.executable, "scripts/01_naive_multimer_inference.py", str(idx), protein1, protein2],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per pair
            )

            if result.returncode == 0:
                logger.info(f"Successfully processed {protein1}-{protein2}")
            else:
                logger.error(f"Failed to process {protein1}-{protein2}")
                logger.error(f"STDERR: {result.stderr}")
                failure_reason = classify_failure(result.stderr)
                failed_pairs.append((idx, protein1, protein2, failure_reason, result.stderr))

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing {protein1}-{protein2}")
            failed_pairs.append((idx, protein1, protein2, "timeout", "Timeout after 1 hour"))
        except Exception as e:
            logger.error(f"Error processing {protein1}-{protein2}: {e}")
            failure_reason = classify_failure(str(e))
            failed_pairs.append((idx, protein1, protein2, failure_reason, str(e)))

    # ── Save failed pairs to CSV ──────────────────────────────
    # if failed_pairs:
    #     # Start from the original ppi_incomplete rows for these pairs
    #     # so all original columns are preserved
    #     failed_indices = [f[0] for f in failed_pairs]
    #     df_failed = ppi_incomplete.loc[ppi_incomplete.index.isin(failed_indices)].copy()

    #     # Build a lookup for the extra columns
    #     reason_lookup = {f[0]: f[3] for f in failed_pairs}
    #     error_lookup  = {f[0]: f[4] for f in failed_pairs}

    #     df_failed["failure_reason"] = df_failed.index.map(reason_lookup)
    #     df_failed["error_details"]  = df_failed.index.map(error_lookup)

    #     failed_csv = base / "reference/1_naive_chai_ppi_sort/ppi_failed.csv"
    #     df_failed.to_csv(failed_csv, index=True, index_label="original_ppi_index")
    #     logger.info(f"Saved {len(df_failed)} failed pairs to {failed_csv}")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL PROCESSING SUMMARY")
    print("="*60)
    print(f"Incomplete pairs loaded       : {len(ppi_incomplete)}")
    print(f"Already complete at start     : {already_complete}")
    print(f"Attempted to process          : {len(pairs_to_run)}")
    print(f"Successfully processed        : {len(pairs_to_run) - len(failed_pairs)}")
    print(f"Failed                        : {len(failed_pairs)}")
    print("="*60)

    if failed_pairs:
        # Count by failure reason
        reason_counts = {}
        for _, _, _, reason, _ in failed_pairs:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("\nFailure breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30}: {count}")
        # print(f"\nFailed pairs saved to: {base / 'reference/1_naive_ppi_sort/ppi_failed.csv'}")

if __name__ == "__main__":
    main()

