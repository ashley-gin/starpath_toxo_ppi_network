#!/usr/bin/env python
# coding: utf-8
"""
Revised monomer processing: expand one row per (protein, crosslink), record
per-model PAE and per-crosslink distances, mean PAE and mean XL distance per
model, and min PAE (across models) plus the distance from that min-PAE model.

Proteins with no crosslink still get one row with PAE scores and null distance columns.

metrics of interest:
  - Per-model and aggregate ptm from scores.model_idx_*.npz: mean across models,
    ptm at the min–mean-off-diagonal-PAE model (pae_best_model_idx), and ptm at
    the min–xl_window_pae_mean model (xl_window_pae_mean_best_model_idx).
  - xl_site_pae: symmetrised PAE at the exact crosslink site,
      mean(PAE[i, j], PAE[j, i]) where i/j are 0-based indices of pos_a/pos_b.
  - xl_window_pae_mean / xl_window_pae_median: mean and median of all PAE values
      in the ±WINDOW_SIZE residue neighbourhood around both crosslink positions
      (off-diagonal pairs only, excluding the exact site itself so that the
      site-level metric and the neighbourhood metric remain independent).
  Both metrics are computed per model and then aggregated (mean across models).
"""

import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, CHAI_MONOMER_DIR, INTRA_XL_CSV, PROCESSED_DATA_DIR #change to INTRA_XL_CSV to NULL_INTRA_XL_CSV if using null intra crosslinks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_CSV = PROCESSED_DATA_DIR / "00_naive_monomer_intra_xl_calculations.csv"
FAILED_CSV = PROCESSED_DATA_DIR / "00_naiev_monomer_intra_xl_failed.csv"

# Half-width of the residue window around each crosslink position (inclusive).
# e.g. WINDOW_SIZE=3 → positions [pos-3, pos-2, ..., pos+3] on each side.
WINDOW_SIZE = 3


def build_monomer_dir_lookup(monomer_dir: Path):
    """
    Scan monomer output directory; map protein_id -> (dir_path, status).
    status is "valid" only if dir has 5 CIF and 5 NPZ.
    """
    pattern = re.compile(r"^monomer_\d+_(TGRH88_\w+)$")
    lookup = {}
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
            lookup[protein_id] = (d, "valid")
        else:
            lookup[protein_id] = (d, "incomplete_files")
    n_valid = sum(1 for _, s in lookup.values() if s == "valid")
    logger.info(f"Monomer dirs: {len(lookup)} total, {n_valid} valid (5 CIF + 5 NPZ)")
    return lookup


def order_models_by_idx(filename):
    """Extract model index from filename."""
    m = re.search(r"model_idx_(\d+)", str(filename))
    return int(m.group(1)) if m else None


def extract_coords_from_cif(cif_path: Path) -> dict:
    """Extract CA coordinates from CIF; keys (chain_id, res_num)."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", str(cif_path))
    coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    res_num = residue.id[1]
                    coords[(chain.id, res_num)] = residue["CA"].coord
    return coords


def get_single_chain_id(coords_dict: dict) -> str:
    """Return the single chain id for a monomer CIF."""
    chain_ids = {k[0] for k in coords_dict}
    if len(chain_ids) != 1:
        raise ValueError(f"Expected one chain, got {chain_ids}")
    return next(iter(chain_ids))


def calculate_distance(coords_dict: dict, chain_id: str, res1: int, res2: int) -> float | None:
    """CA-CA distance between two residues on the same chain."""
    c1 = coords_dict.get((chain_id, res1))
    c2 = coords_dict.get((chain_id, res2))
    if c1 is None or c2 is None:
        return None
    return float(np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2)))


def mean_off_diagonal_pae(pae: np.ndarray) -> float:
    """Mean PAE excluding the diagonal (single NxN monomer matrix)."""
    n = pae.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(pae[mask].mean())


def extract_ptm_from_npz(npz_path: Path) -> float | None:
    """
    Load a Chai scores .npz and return the predicted TM score (ptm), or None
    if the key is missing or the file cannot be read.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "ptm" not in data.files:
                logger.warning(f"{npz_path.name}: no 'ptm' key in NPZ")
                return None
            return float(np.asarray(data["ptm"]).squeeze())
    except Exception as e:
        logger.warning(f"{npz_path.name} PTM: {e}")
        return None


def _ptm_from_open_npz(data: np.lib.npyio.NpzFile) -> float | None:
    """Scalar ptm from an already-open scores NPZ (avoids a second disk read)."""
    try:
        if "ptm" not in data.files:
            return None
        return float(np.asarray(data["ptm"]).squeeze())
    except Exception as e:
        logger.warning(f"PTM from open NPZ: {e}")
        return None


def extract_crosslink_pae_metrics(
    pae: np.ndarray,
    pos_a: int,
    pos_b: int,
    window: int = WINDOW_SIZE,
) -> dict:
    """
    Compute site-specific and windowed PAE metrics for a single crosslink on a
    single model's PAE matrix.

    Parameters
    ----------
    pae : np.ndarray, shape (N, N)
        PAE matrix for one model. Row i = "score for residue i given residue j
        as the reference frame" (Chai / AF2 convention).
        **Indices are 0-based; crosslink positions are 1-based and converted
        internally.**
    pos_a, pos_b : int
        1-based residue positions of the crosslink (as stored in the CSV).
    window : int
        Half-width of the neighbourhood window (default WINDOW_SIZE = 3),
        so the window spans [pos - window, pos + window] inclusive on each side.

    Returns
    -------
    dict with keys:
      xl_site_pae
          Symmetrised PAE at the exact crosslink pair:
          mean(PAE[i, j], PAE[j, i]).  None if either index is out of range.
      xl_window_pae_mean
          Mean of all unique off-diagonal PAE values in the rectangular
          window region defined by the ±window neighbourhood of pos_a and pos_b.
          The exact site pair (i,j) and (j,i) are included; purely diagonal
          entries (i == j) are excluded.  None if no valid entries exist.
      xl_window_pae_median
          Median of the same window values.  None if no valid entries exist.

    Window construction
    -------------------
    Let rows = [max(0, i-w) .. min(N-1, i+w)] for the row dimension (anchored
    on pos_a), and cols = [max(0, j-w) .. min(N-1, j+w)] for the column
    dimension (anchored on pos_b).  We collect PAE[r, c] for all (r, c) in
    rows × cols where r ≠ c, *plus* the symmetric counterpart block
    PAE[c, r] so that the window is symmetric around both positions.
    This gives a neighbourhood that spans both the (A→B reference) and
    (B→A reference) halves of the PAE matrix.
    """
    n = pae.shape[0]

    # Convert to 0-based indices
    i = pos_a - 1
    j = pos_b - 1

    result: dict = {
        "xl_site_pae": None,
        "xl_window_pae_mean": None,
        "xl_window_pae_median": None,
    }

    # Bounds check for the exact site
    if not (0 <= i < n and 0 <= j < n):
        logger.debug(f"Crosslink positions ({pos_a}, {pos_b}) out of PAE range (N={n}); skipping site PAE.")
        return result

    # --- Exact site PAE (symmetrised) ---
    result["xl_site_pae"] = float(np.mean([pae[i, j], pae[j, i]]))

    # --- Windowed PAE ---
    # Row indices centred on pos_a; col indices centred on pos_b.
    # We also collect the transposed block (centred on pos_b rows / pos_a cols)
    # so the metric reflects both directions of the PAE matrix.
    rows_a = np.arange(max(0, i - window), min(n, i + window + 1))  # around pos_a
    cols_b = np.arange(max(0, j - window), min(n, j + window + 1))  # around pos_b

    # Build index grids for both orientations
    r_grid_ab, c_grid_ab = np.meshgrid(rows_a, cols_b, indexing="ij")  # (A-rows, B-cols)
    r_grid_ba, c_grid_ba = np.meshgrid(cols_b, rows_a, indexing="ij")  # (B-rows, A-cols)

    # Stack both orientations and flatten
    all_r = np.concatenate([r_grid_ab.ravel(), r_grid_ba.ravel()])
    all_c = np.concatenate([c_grid_ab.ravel(), c_grid_ba.ravel()])

    # Remove diagonal (self-reference residues have no meaningful PAE)
    off_diag = all_r != all_c
    all_r = all_r[off_diag]
    all_c = all_c[off_diag]

    if all_r.size == 0:
        return result

    window_vals = pae[all_r, all_c]
    result["xl_window_pae_mean"] = float(np.mean(window_vals))
    result["xl_window_pae_median"] = float(np.median(window_vals))

    return result


def expand_monomer_with_crosslinks(valid_protein_ids: list, xl_rows_by_protein: dict) -> list[dict]:
    """
    Build one row per (protein, crosslink). If a protein has no crosslinks,
    add one row with Crosslink Position A/B and residue columns as None.
    xl_rows_by_protein: {protein_id: [{"Crosslink Position A": int, "Crosslink Position B": int,
                                       "Crosslinked Residue A": _, "Crosslinked Residue B": _}, ...]}
    """
    expanded = []
    for protein_id in valid_protein_ids:
        rows = xl_rows_by_protein.get(protein_id, [])
        if not rows:
            expanded.append({
                "protein_id": protein_id,
                "Crosslink Position A": None,
                "Crosslink Position B": None,
                "Crosslinked Residue A": None,
                "Crosslinked Residue B": None,
            })
        else:
            for xl in rows:
                expanded.append({
                    "protein_id": protein_id,
                    "Crosslink Position A": xl["Crosslink Position A"],
                    "Crosslink Position B": xl["Crosslink Position B"],
                    "Crosslinked Residue A": xl.get("Crosslinked Residue A"),
                    "Crosslinked Residue B": xl.get("Crosslinked Residue B"),
                })
    logger.info(f"Expanded to {len(expanded)} rows (one per protein–crosslink or protein with no XL)")
    return expanded


def load_intra_crosslinks_expanded(csv_path: Path) -> tuple[dict[str, list], list[str]]:
    """
    Load intra crosslinks CSV. Return:
    - xl_rows_by_protein: {protein_id: [{"Crosslink Position A", "Crosslink Position B",
                                         "Crosslinked Residue A", "Crosslinked Residue B"}, ...]}
    - unique_proteins: list of protein IDs that have at least one crosslink (for reference).
    """
    df = pd.read_csv(csv_path)
    intra = df[df["Leading Protein A"] == df["Leading Protein B"]].copy()
    pos_a_col = "Crosslink Position A"
    pos_b_col = "Crosslink Position B"
    protein_col = "Leading Protein A"
    res_a_col = "Crosslinked Residue A"
    res_b_col = "Crosslinked Residue B"
    intra = intra.dropna(subset=[protein_col, pos_a_col, pos_b_col])
    intra[pos_a_col] = intra[pos_a_col].astype(int)
    intra[pos_b_col] = intra[pos_b_col].astype(int)

    by_protein = {}
    for _, row in intra.iterrows():
        pid = row[protein_col]
        entry = {
            "Crosslink Position A": int(row[pos_a_col]),
            "Crosslink Position B": int(row[pos_b_col]),
            "Crosslinked Residue A": row.get(res_a_col),
            "Crosslinked Residue B": row.get(res_b_col),
        }
        by_protein.setdefault(pid, []).append(entry)
    # Deduplicate by (pos_a, pos_b) per protein
    for pid in by_protein:
        seen = set()
        unique = []
        for xl in by_protein[pid]:
            key = (xl["Crosslink Position A"], xl["Crosslink Position B"])
            if key not in seen:
                seen.add(key)
                unique.append(xl)
        by_protein[pid] = unique
    logger.info(f"Intra crosslinks: {intra.shape[0]} rows, {len(by_protein)} proteins with ≥1 crosslink")
    return by_protein, list(by_protein.keys())


def process_monomer_directory_raw(monomer_dir: Path) -> dict | None:
    """
    Load all 5 models for one monomer dir: per-model mean off-diagonal PAE,
    (coords, chain_id) per model, and the raw PAE matrix per model so that
    site-specific metrics can be computed for any crosslink without re-reading.

    Returns dict with:
      model_0_pae .. model_4_pae   – mean off-diagonal PAE (scalar)
      model_0_ptm .. model_4_ptm   – ptm from each scores NPZ (or None)
      model_0_pae_matrix .. model_4_pae_matrix  – raw NxN np.ndarray (or None)
      models_data  – list of 5 tuples (coords_dict, chain_id)
    """
    cif_files = sorted(monomer_dir.glob("pred.model_idx_*.cif"), key=order_models_by_idx)
    npz_files = sorted(monomer_dir.glob("scores.model_idx_*.npz"), key=order_models_by_idx)
    if len(cif_files) != 5 or len(npz_files) != 5:
        return None

    out = {}
    models_data = []

    for model_idx, (cif_path, npz_path) in enumerate(zip(cif_files, npz_files)):
        coords, chain_id = None, None
        try:
            coords = extract_coords_from_cif(cif_path)
            chain_id = get_single_chain_id(coords)
        except Exception as e:
            logger.warning(f"{monomer_dir.name} model {model_idx} CIF: {e}")
        models_data.append((coords, chain_id))

        pae_val = None
        pae_matrix = None
        ptm_val = None
        try:
            data = np.load(npz_path, allow_pickle=False)
            pae_matrix = data["pae"]                    # keep the full matrix
            pae_val = mean_off_diagonal_pae(pae_matrix)
            ptm_val = _ptm_from_open_npz(data)
        except Exception as e:
            logger.warning(f"{monomer_dir.name} model {model_idx} NPZ: {e}")

        out[f"model_{model_idx}_pae"] = pae_val
        out[f"model_{model_idx}_ptm"] = ptm_val
        out[f"model_{model_idx}_pae_matrix"] = pae_matrix  # None if load failed

    if not any(out.get(f"model_{i}_pae") for i in range(5)):
        return None
    out["models_data"] = models_data
    return out


def build_row_from_raw(
    raw: dict,
    pos_a: int | None,
    pos_b: int | None,
) -> dict:
    """
    From raw per-model PAE scalars, PAE matrices, and models_data, compute:
      - per-model distances
      - distance aggregates (mean/std/min/max)
      - mean off-diagonal PAE across models
      - min-PAE model index, value, and its crosslink distance
      - per-model ptm, mean ptm, ptm at pae_best / xl_window_pae_mean_best models
      - per-model site PAE, window mean, window median  (new)
      - mean across models of each site/window metric    (new)
    """
    models_data = raw["models_data"]
    paes = [raw.get(f"model_{i}_pae") for i in range(5)]
    ptms = [raw.get(f"model_{i}_ptm") for i in range(5)]
    pae_matrices = [raw.get(f"model_{i}_pae_matrix") for i in range(5)]

    distances = []
    for i in range(5):
        coords, chain_id = models_data[i]
        if (pos_a is not None and pos_b is not None) and coords is not None and chain_id is not None:
            d = calculate_distance(coords, chain_id, pos_a, pos_b)
            distances.append(d)
        else:
            distances.append(None)

    # --- Per-model columns (existing) ---
    result = {}
    for i in range(5):
        result[f"model_{i}_pae"] = paes[i]
        result[f"model_{i}_ptm"] = ptms[i]
        result[f"model_{i}_distance"] = distances[i] if i < len(distances) else None

    # --- Aggregate distances ---
    valid_dists = [d for d in distances if d is not None]
    if valid_dists:
        result["distance_mean"] = float(np.mean(valid_dists))
        result["distance_std"] = float(np.std(valid_dists, ddof=1)) if len(valid_dists) > 1 else None
        result["distance_min"] = float(np.min(valid_dists))
        result["distance_max"] = float(np.max(valid_dists))
    else:
        result["distance_mean"] = None
        result["distance_std"] = None
        result["distance_min"] = None
        result["distance_max"] = None

    # --- Mean off-diagonal PAE across models ---
    valid_paes = [p for p in paes if p is not None]
    result["pae_mean"] = float(np.mean(valid_paes)) if valid_paes else None

    # --- Mean ptm across models ---
    valid_ptms = [p for p in ptms if p is not None]
    result["ptm_mean"] = float(np.mean(valid_ptms)) if valid_ptms else None

    # --- Min-PAE model ---
    valid_pae_idxs = [(i, paes[i]) for i in range(5) if paes[i] is not None]
    if not valid_pae_idxs:
        result["pae_best_model_idx"] = None
        result["pae_best_model"] = None
        result["pae_best_model_distance"] = None
        result["ptm_pae_best_model"] = None
    else:
        min_idx, min_pae = min(valid_pae_idxs, key=lambda x: x[1])
        result["pae_best_model_idx"] = min_idx
        result["pae_best_model"] = min_pae
        result["pae_best_model_distance"] = distances[min_idx] if min_idx < len(distances) else None
        result["ptm_pae_best_model"] = ptms[min_idx] if min_idx < len(ptms) else None

    # --- Site-specific and windowed PAE (new) ---
    # Only meaningful when we have a crosslink to look up.
    per_model_site: list[float | None] = []
    per_model_window_mean: list[float | None] = []
    per_model_window_median: list[float | None] = []

    for i in range(5):
        pae_matrix = pae_matrices[i]
        if pos_a is not None and pos_b is not None and pae_matrix is not None:
            xl_metrics = extract_crosslink_pae_metrics(pae_matrix, pos_a, pos_b)
            per_model_site.append(xl_metrics["xl_site_pae"])
            per_model_window_mean.append(xl_metrics["xl_window_pae_mean"])
            per_model_window_median.append(xl_metrics["xl_window_pae_median"])
        else:
            per_model_site.append(None)
            per_model_window_mean.append(None)
            per_model_window_median.append(None)

        result[f"model_{i}_xl_site_pae"] = per_model_site[-1]
        result[f"model_{i}_xl_window_pae_mean"] = per_model_window_mean[-1]
        result[f"model_{i}_xl_window_pae_median"] = per_model_window_median[-1]

    # Aggregate site / window metrics across models
    valid_site = [v for v in per_model_site if v is not None]
    valid_win_mean = [v for v in per_model_window_mean if v is not None]
    valid_win_med = [v for v in per_model_window_median if v is not None]

    result["xl_site_pae_mean"] = float(np.mean(valid_site)) if valid_site else None
    result["xl_window_pae_mean_mean"] = float(np.mean(valid_win_mean)) if valid_win_mean else None
    result["xl_window_pae_median_mean"] = float(np.mean(valid_win_med)) if valid_win_med else None

    # --- Min xl_site_pae model ---
    valid_site_idxs = [(i, per_model_site[i]) for i in range(5) if per_model_site[i] is not None]
    if not valid_site_idxs:
        result["xl_site_pae_best_model_idx"] = None
        result["xl_site_pae_best_model"] = None
        result["xl_site_pae_best_model_distance"] = None
    else:
        min_site_idx, min_site_val = min(valid_site_idxs, key=lambda x: x[1])
        result["xl_site_pae_best_model_idx"] = min_site_idx
        result["xl_site_pae_best_model"] = min_site_val
        result["xl_site_pae_best_model_distance"] = distances[min_site_idx]

    # --- Min xl_window_pae_mean model ---
    valid_win_mean_idxs = [(i, per_model_window_mean[i]) for i in range(5) if per_model_window_mean[i] is not None]
    if not valid_win_mean_idxs:
        result["xl_window_pae_mean_best_model_idx"] = None
        result["xl_window_pae_mean_best_model"] = None
        result["xl_window_pae_mean_best_model_distance"] = None
        result["ptm_xl_window_pae_mean_best_model"] = None
    else:
        min_win_idx, min_win_val = min(valid_win_mean_idxs, key=lambda x: x[1])
        result["xl_window_pae_mean_best_model_idx"] = min_win_idx
        result["xl_window_pae_mean_best_model"] = min_win_val
        result["xl_window_pae_mean_best_model_distance"] = distances[min_win_idx]
        result["ptm_xl_window_pae_mean_best_model"] = (
            ptms[min_win_idx] if min_win_idx < len(ptms) else None
        )

    return result


def main():
    logger.info("Starting revised monomer intra-XL PAE & distance pipeline (one row per protein–crosslink)")
    dir_lookup = build_monomer_dir_lookup(CHAI_MONOMER_DIR)
    xl_rows_by_protein, _ = load_intra_crosslinks_expanded(INTRA_XL_CSV)
    valid_protein_ids = [pid for pid, (_, s) in dir_lookup.items() if s == "valid"]
    df_expanded = expand_monomer_with_crosslinks(valid_protein_ids, xl_rows_by_protein)

    valid_rows = []
    failed_rows = []
    protein_cache = {}

    for idx, row in enumerate(df_expanded):
        protein_id = row["protein_id"]
        pos_a = row.get("Crosslink Position A")
        pos_b = row.get("Crosslink Position B")
        if (idx + 1) % 500 == 0:
            logger.info(f"Processing {idx + 1}/{len(df_expanded)}")

        dir_entry = dir_lookup.get(protein_id)
        if dir_entry is None:
            failed_rows.append({**row, "failure_reason": "no_directory_found"})
            continue
        monomer_dir, status = dir_entry
        if status != "valid":
            failed_rows.append({**row, "failure_reason": status})
            continue

        # Cache raw result per protein (PAE + coords per model)
        if protein_id not in protein_cache:
            raw = process_monomer_directory_raw(monomer_dir)
            if raw is None:
                failed_rows.append({**row, "failure_reason": "no_pae_or_models"})
                continue
            protein_cache[protein_id] = raw

        raw = protein_cache[protein_id]
        pos_a_int = int(pos_a) if pd.notna(pos_a) and pos_a is not None else None
        pos_b_int = int(pos_b) if pd.notna(pos_b) and pos_b is not None else None
        metrics = build_row_from_raw(raw, pos_a_int, pos_b_int)

        if metrics["pae_mean"] is None and metrics["distance_mean"] is None:
            failed_rows.append({**row, "failure_reason": "no_metrics"})
            continue

        out_row = {**row, **metrics}
        valid_rows.append(out_row)

    if valid_rows:
        df_out = pd.DataFrame(valid_rows)
        df_out.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved {len(valid_rows)} rows to {OUTPUT_CSV}")
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_CSV, index=False)
        logger.info(f"Saved {len(failed_rows)} failed to {FAILED_CSV}")
    logger.info("Done.")


if __name__ == "__main__":
    main()