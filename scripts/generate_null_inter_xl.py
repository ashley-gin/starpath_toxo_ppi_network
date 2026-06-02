#!/usr/bin/env python
# coding: utf-8
"""
Generate a null inter-crosslink dataset by randomly pairing lysine residues
BETWEEN two protein chains in each multimer.

For each protein pair in the multimer directory:
  1. Look up both sequences in the proteome mapping.
  2. Find all lysine (K) positions (1-based) in each chain.
  3. Randomly sample K-K pairs across chains (one K from chain A, one from chain B),
     matching the count from the real inter-crosslink dataset for that pair.
  4. Write a CSV in the same format as inter_crosslinks_unique.csv.

Output columns:
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
BASE_DIR          = Path("/home/ubuntu/chai-lab/updated_analyses_SL")
CHAI_MULTIMER_DIR = BASE_DIR / "chai_outputs/naive_chai_ppis_noXLs"
INTER_XL_CSV      = BASE_DIR / "reference/rh88_crosslinks/inter_crosslinks_unique.csv"
PROTEOME_CSV      = BASE_DIR / "reference/rh88_proteome/ToxoDB_TGRH88_Protein_Sequences.csv"
OUTPUT_CSV        = BASE_DIR / "reference/rh88_crosslinks/null_inter_crosslinks_random_lys.csv"

# ── Options ────────────────────────────────────────────────────────────────────
PROTEOME_ID_COL  = "Gene_ID"
PROTEOME_SEQ_COL = "Protein_Sequence"

# How many random pairs to generate per protein pair.
# "match_real"  → same count as the real dataset for that protein pair (default)
# "all_pairs"   → every possible K (chain A) × K (chain B) combination
# integer N     → exactly N pairs (skipped if either chain has no lysines)
PAIRING_MODE: str | int = "match_real"

RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_multimer_protein_pairs(multimer_dir: Path) -> list[tuple[str, str]]:
    """
    Return (protein_id_A, protein_id_B) for all valid multimer directories.
    Expected format: pair_<N>_<TGRH88_XXXXXX>_<TGRH88_XXXXXX>
    e.g. pair_0_TGRH88_000140_TGRH88_035710
    """
    pattern = re.compile(r"^pair_\d+_(TGRH88_\w+)_(TGRH88_\w+)$")

    valid = []
    for d in multimer_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue

        protein_a, protein_b = m.group(1), m.group(2)

        cif_files = list(d.glob("pred.model_idx_*.cif"))
        npz_files = list(d.glob("scores.model_idx_*.npz"))
        if len(cif_files) == 5 and len(npz_files) == 5:
            valid.append((protein_a, protein_b))

    logger.info(f"Found {len(valid)} valid multimer protein pairs")
    return valid


def create_proteome_mapping(df_proteome: pd.DataFrame,
                            id_col: str  = PROTEOME_ID_COL,
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
    """Return 1-based positions of all lysine (K) residues in a sequence."""
    return [i + 1 for i, aa in enumerate(sequence) if aa == "K"]


def load_real_inter_xl_counts(xl_csv: Path) -> dict[tuple[str, str], int]:
    """
    Return {(protein_A, protein_B): n_crosslinks} from the real inter-crosslink CSV.
    Only rows where Leading Protein A != Leading Protein B are considered.
    The pair key is stored in the order they appear in the CSV (A, B).
    """
    df = pd.read_csv(xl_csv)
    inter = df[df["Leading Protein A"] != df["Leading Protein B"]].copy()
    inter = inter.dropna(subset=[
        "Leading Protein A", "Leading Protein B",
        "Crosslink Position A", "Crosslink Position B"
    ])
    inter = inter.drop_duplicates(subset=[
        "Leading Protein A", "Leading Protein B",
        "Crosslink Position A", "Crosslink Position B"
    ])
    counts = (
        inter.groupby(["Leading Protein A", "Leading Protein B"])
        .size()
        .to_dict()         # keys are (protein_A, protein_B) tuples
    )
    logger.info(f"Real inter-XL counts loaded for {len(counts)} protein pairs")
    return counts


def random_inter_lys_pairs(lys_a: list[int], lys_b: list[int],
                            n_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    """
    Draw n_pairs unique (pos_a, pos_b) combinations where pos_a ∈ lys_a
    and pos_b ∈ lys_b (cross-chain — no within-chain constraint needed).
    If n_pairs >= total available combinations, return all of them.
    """
    all_pairs = [(a, b) for a in lys_a for b in lys_b]
    if n_pairs >= len(all_pairs):
        return all_pairs
    return rng.sample(all_pairs, n_pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)

    protein_pairs  = build_multimer_protein_pairs(CHAI_MULTIMER_DIR)
    df_proteome    = pd.read_csv(PROTEOME_CSV)
    seq_map        = create_proteome_mapping(df_proteome)
    real_xl_counts = load_real_inter_xl_counts(INTER_XL_CSV) if PAIRING_MODE == "match_real" else {}

    rows = []
    skipped_no_seq      = []
    skipped_no_lys      = []
    skipped_not_in_real = 0

    for protein_a, protein_b in protein_pairs:
        seq_a = seq_map.get(protein_a)
        seq_b = seq_map.get(protein_b)

        if seq_a is None or seq_b is None:
            missing = [p for p, s in [(protein_a, seq_a), (protein_b, seq_b)] if s is None]
            skipped_no_seq.extend(missing)
            continue

        lys_a = lysine_positions(seq_a)
        lys_b = lysine_positions(seq_b)

        if not lys_a or not lys_b:
            skipped_no_lys.append((protein_a, protein_b))
            continue

        # Determine n_pairs ───────────────────────────────────────────────────
        if PAIRING_MODE == "match_real":
            # Try both orientations since CSV may list the pair in either order
            n_pairs = real_xl_counts.get((protein_a, protein_b)) or \
                      real_xl_counts.get((protein_b, protein_a))
            if n_pairs is None:
                skipped_not_in_real += 1
                continue
        elif PAIRING_MODE == "all_pairs":
            n_pairs = len(lys_a) * len(lys_b)
        elif isinstance(PAIRING_MODE, int):
            n_pairs = PAIRING_MODE
        else:
            raise ValueError(f"Unknown PAIRING_MODE: {PAIRING_MODE!r}")

        pairs = random_inter_lys_pairs(lys_a, lys_b, n_pairs, rng)

        for pos_a, pos_b in pairs:
            rows.append({
                "Leading Protein A":     protein_a,
                "Leading Protein B":     protein_b,
                "Crosslink Position A":  pos_a,
                "Crosslink Position B":  pos_b,
                "Crosslinked Residue A": "K",
                "Crosslinked Residue B": "K",
            })

    df_out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    logger.info(
        f"Wrote {len(df_out)} null inter-crosslink rows "
        f"for {df_out[['Leading Protein A', 'Leading Protein B']].drop_duplicates().shape[0]} "
        f"protein pairs → {OUTPUT_CSV}"
    )
    if skipped_no_seq:
        logger.warning(f"No sequence found for {len(skipped_no_seq)} proteins: {skipped_no_seq[:5]} ...")
    if skipped_no_lys:
        logger.warning(f"No lysines in one or both chains for {len(skipped_no_lys)} pairs: {skipped_no_lys[:3]} ...")
    if skipped_not_in_real:
        logger.warning(f"Skipped {skipped_not_in_real} pairs not found in the real inter-XL dataset")


if __name__ == "__main__":
    main()