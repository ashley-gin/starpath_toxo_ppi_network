#!/usr/bin/env python
# coding: utf-8
"""
Generate a null intra-crosslink dataset by randomly pairing lysine residues
within each protein sequence.

For each protein in the monomer directory:
  1. Look up its sequence in the proteome mapping.
  2. Find all lysine (K) positions (1-based).
  3. Randomly pair them without replacement, producing the same number of
     crosslinks as the real dataset for that protein (or a fixed number if
     the real count is unavailable / you prefer a fixed n).
  4. Write a CSV in the same format as intra_crosslinks_unique.csv so it
     can be dropped in as INTRA_XL_CSV in 03b2_processing_naive_chai_monomers_minPAE.py.

Output columns match the real crosslink CSV:
  Leading Protein A, Leading Protein B,
  Crosslink Position A, Crosslink Position B,
  Crosslinked Residue A, Crosslinked Residue B   (always K, K)
"""

import logging
import random
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path("/home/ubuntu/chai-lab/updated_analyses_SL")
CHAI_MONOMER_DIR = BASE_DIR / "chai_outputs/naive_chai_monomers_noXLs"
INTRA_XL_CSV    = BASE_DIR / "reference/rh88_crosslinks/intra_crosslinks_unique.csv"
PROTEOME_CSV    = BASE_DIR / "reference/rh88_proteome/ToxoDB_TGRH88_Protein_Sequences.csv"
OUTPUT_CSV      = BASE_DIR / "reference/rh88_crosslinks/null_intra_crosslinks_random_lys.csv"

# ── Options ────────────────────────────────────────────────────────────────────
PROTEOME_ID_COL  = "Gene_ID"
PROTEOME_SEQ_COL = "Protein_Sequence"

# How many random pairs to generate per protein.
# "match_real"  → same count as the real dataset for that protein (default)
# "all_pairs"   → every possible unordered K-K pair
# integer N     → exactly N pairs (skipped if fewer than 2 lysines available)
PAIRING_MODE: str | int = "match_real"

RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_monomer_protein_ids(monomer_dir: Path) -> list[str]:
    """Return protein IDs for all valid monomer directories (5 CIF + 5 NPZ)."""
    pattern = re.compile(r"^monomer_\d+_(TGRH88_\w+)$")
    valid = []
    for d in monomer_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        protein_id = m.group(1)
        cif_files = list(d.glob("pred.model_idx_*.cif"))
        npz_files = list(d.glob("scores.model_idx_*.npz"))
        if len(cif_files) == 5 and len(npz_files) == 5:
            valid.append(protein_id)
    logger.info(f"Found {len(valid)} valid monomer protein IDs")
    return valid


def create_proteome_mapping(df_proteome: pd.DataFrame,
                            id_col: str = PROTEOME_ID_COL,
                            seq_col: str = PROTEOME_SEQ_COL) -> dict:
    """Create a Gene_ID -> sequence mapping from the proteome CSV."""
    duplicates = df_proteome[df_proteome.duplicated(subset=id_col, keep=False)]
    if not duplicates.empty:
        logger.warning(
            f"Found {duplicates[id_col].nunique()} duplicate Gene_ID(s) — keeping first occurrence."
        )
    mapping = (
        df_proteome.drop_duplicates(subset=id_col, keep="first")
        .set_index(id_col)[seq_col]
        .to_dict()
    )
    logger.info(f"Created mapping for {len(mapping)} proteins")
    return mapping


def lysine_positions(sequence: str) -> list[int]:
    """Return 1-based positions of all lysine (K) residues in the sequence."""
    return [i + 1 for i, aa in enumerate(sequence) if aa == "K"]


def load_real_xl_counts(xl_csv: Path) -> dict[str, int]:
    """
    Return {protein_id: n_crosslinks} from the real intra-crosslink CSV.
    Used when PAIRING_MODE == "match_real".
    """
    df = pd.read_csv(xl_csv)
    intra = df[df["Leading Protein A"] == df["Leading Protein B"]].copy()
    intra = intra.dropna(subset=["Leading Protein A", "Crosslink Position A", "Crosslink Position B"])
    counts = (
        intra.drop_duplicates(subset=["Leading Protein A", "Crosslink Position A", "Crosslink Position B"])
        .groupby("Leading Protein A")
        .size()
        .to_dict()
    )
    logger.info(f"Real XL counts loaded for {len(counts)} proteins")
    return counts


def random_lys_pairs(lys_pos: list[int], n_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """
    Draw n_pairs unique unordered pairs from lys_pos without replacement.
    If n_pairs exceeds the number of available pairs, return all pairs.
    Pairs are returned as (smaller_pos, larger_pos).
    """
    all_pairs = [
        (a, b)
        for idx, a in enumerate(lys_pos)
        for b in lys_pos[idx + 1:]
    ]
    if n_pairs >= len(all_pairs):
        return all_pairs
    return rng.sample(all_pairs, n_pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)

    protein_ids   = build_monomer_protein_ids(CHAI_MONOMER_DIR)
    df_proteome   = pd.read_csv(PROTEOME_CSV)
    seq_map       = create_proteome_mapping(df_proteome)
    real_xl_counts = load_real_xl_counts(INTRA_XL_CSV) if PAIRING_MODE == "match_real" else {}

    rows = []
    skipped_no_seq      = []
    skipped_too_few_lys = []

    for protein_id in protein_ids:
        sequence = seq_map.get(protein_id)
        if sequence is None:
            skipped_no_seq.append(protein_id)
            continue

        lys_pos = lysine_positions(sequence)
        if len(lys_pos) < 2:
            skipped_too_few_lys.append(protein_id)
            continue

        # Determine how many pairs to draw
        if PAIRING_MODE == "match_real":
            # n_pairs = real_xl_counts.get(protein_id, 1)  # default 1 if not in real dataset
            #updated with code below to match same number of crosslinks in the real dataset
            n_pairs = real_xl_counts.get(protein_id)
            if n_pairs is None:
                continue #skip protein if no crosslinks in the real dataset
        elif PAIRING_MODE == "all_pairs":
            n_pairs = len(lys_pos) * (len(lys_pos) - 1) // 2
        elif isinstance(PAIRING_MODE, int):
            n_pairs = PAIRING_MODE
        else:
            raise ValueError(f"Unknown PAIRING_MODE: {PAIRING_MODE!r}")

        pairs = random_lys_pairs(lys_pos, n_pairs, rng)

        for pos_a, pos_b in pairs:
            rows.append({
                "Leading Protein A":    protein_id,
                "Leading Protein B":    protein_id,   # intra = same protein
                "Crosslink Position A": pos_a,
                "Crosslink Position B": pos_b,
                "Crosslinked Residue A": "K",
                "Crosslinked Residue B": "K",
            })

    df_out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"Wrote {len(df_out)} null crosslink rows for {df_out['Leading Protein A'].nunique()} proteins → {OUTPUT_CSV}")
    if skipped_no_seq:
        logger.warning(f"No sequence found for {len(skipped_no_seq)} proteins: {skipped_no_seq[:5]} ...")
    if skipped_too_few_lys:
        logger.warning(f"Fewer than 2 lysines in {len(skipped_too_few_lys)} proteins: {skipped_too_few_lys[:5]} ...")


if __name__ == "__main__":
    main()