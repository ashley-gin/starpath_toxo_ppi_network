#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
from pathlib import Path
import os
import logging
import numpy as np
import re
from Bio.PDB import MMCIFParser
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT, PROTEOME_CSV, PPI_CSV, CHAI_HOMODIMER_DIR, INTRA_XL_CSV, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_CSV = PROCESSED_DATA_DIR / "02_naive_homodimer_intra_xl_calculations.csv"
FAILED_CSV = PROCESSED_DATA_DIR / "02_naive_homodimer_intra_xl_failed.csv"

# Half-width of the residue window around each crosslink position (inclusive).
# e.g. WINDOW_SIZE=3 → positions [pos-3, ..., pos+3] on each side.
# Mirrors WINDOW_SIZE in 03b3_processing_naive_chai_monomers_intraXL_window.py.
WINDOW_SIZE = 3

def load_data():
    """Load all required data files."""
    try:
        df_ppi = pd.read_csv(PPI_CSV)
        logger.info(f"Loaded {len(df_ppi)} PPIs from combined_PPI.csv")

        df_proteome = pd.read_csv(PROTEOME_CSV)
        logger.info(f"Loaded {len(df_proteome)} proteins from proteome")

        df_xlms = pd.read_csv(INTRA_XL_CSV)
        logger.info(f"Loaded {len(df_xlms)} crosslinks from XL-MS data")

        return df_ppi, df_proteome, df_xlms

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_proteome_mapping(df_proteome,
                            id_col="Gene_ID",
                            seq_col="Protein_Sequence") -> dict:
    """Create a Gene_ID -> sequence mapping from the proteome CSV."""
    duplicates = df_proteome[df_proteome.duplicated(subset=id_col, keep=False)]
    if not duplicates.empty:
        logger.warning(f"Found {duplicates[id_col].nunique()} duplicate Gene_ID(s) — keeping first occurrence.")

    mapping = (
        df_proteome.drop_duplicates(subset=id_col, keep="first")
        .set_index(id_col)[seq_col]
        .to_dict()
    )
    logger.info(f"Created mapping for {len(mapping)} proteins")
    return mapping


def build_homodimer_ppi_self_pairs(df_ppi_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Build one PPI row per unique accession with Protein A == Protein B == id.

    Mirrors ``get_unique_proteins`` in ``multi_inference_homodimers_chai_20260424.py``:
    concatenate ``Protein A`` and ``Protein B`` from the combined PPI table, take
    uniques, sort — so ``pair_dir_lookup`` keys ``(id, id)`` match Chai homodimer
    output layout.
    """
    all_proteins = pd.concat(
        [df_ppi_combined["Protein A"], df_ppi_combined["Protein B"]],
        ignore_index=True,
    )
    unique_proteins = all_proteins.drop_duplicates().sort_values().reset_index(drop=True)

    cols = df_ppi_combined.columns.tolist()
    rows = []
    for pid in unique_proteins:
        row = {c: np.nan for c in cols}
        row["Protein A"] = pid
        row["Protein B"] = pid
        rows.append(row)

    df_homo = pd.DataFrame(rows, columns=cols)
    logger.info(
        f"Built {len(df_homo)} homodimer self-pair rows from {len(df_ppi_combined)} "
        f"combined PPI rows ({len(unique_proteins)} unique accessions)"
    )
    return df_homo


def expand_ppi_with_crosslinks(df_ppi, df_xlms):
    """
    Expand PPI dataframe to include crosslink information.
    Creates multiple rows for PPIs with multiple crosslinks.
    """
    expanded_rows = []

    for _, ppi_row in df_ppi.iterrows():
        protein_a = ppi_row['Protein A']
        protein_b = ppi_row['Protein B']

        crosslinks = df_xlms[
            ((df_xlms['Leading Protein A'] == protein_a) &
             (df_xlms['Leading Protein B'] == protein_b)) |
            ((df_xlms['Leading Protein A'] == protein_b) &
             (df_xlms['Leading Protein B'] == protein_a))
        ]

        if len(crosslinks) == 0:
            expanded_row = ppi_row.to_dict()
            expanded_row.update({
                'Crosslink Position A': None,
                'Crosslink Position B': None,
                'Crosslinked Residue A': None,
                'Crosslinked Residue B': None
            })
            expanded_rows.append(expanded_row)
        else:
            for _, xl_row in crosslinks.iterrows():
                expanded_row = ppi_row.to_dict()
                if xl_row['Leading Protein A'] == protein_a:
                    expanded_row.update({
                        'Crosslink Position A': xl_row['Crosslink Position A'],
                        'Crosslink Position B': xl_row['Crosslink Position B'],
                        'Crosslinked Residue A': xl_row['Crosslinked Residue A'],
                        'Crosslinked Residue B': xl_row['Crosslinked Residue B']
                    })
                else:
                    expanded_row.update({
                        'Crosslink Position A': xl_row['Crosslink Position B'],
                        'Crosslink Position B': xl_row['Crosslink Position A'],
                        'Crosslinked Residue A': xl_row['Crosslinked Residue B'],
                        'Crosslinked Residue B': xl_row['Crosslinked Residue A']
                    })
                expanded_rows.append(expanded_row)

    df_expanded = pd.DataFrame(expanded_rows)
    logger.info(f"Expanded {len(df_ppi)} PPIs to {len(df_expanded)} rows with crosslink info")
    return df_expanded


def build_pair_dir_lookup(chai_outputs_dir):
    """
    Pre-scan the output directory once and build a
    (protein_a, protein_b) -> (dir, status) lookup dict.
    Keeps the lowest-indexed directory if multiple matches exist for a pair.
    """
    candidates = {}

    for d in chai_outputs_dir.iterdir():
        if not d.is_dir():
            continue
        match = re.match(r'homodimer_(\d+)_(TGRH88_\d+)$', d.name) #d+ checks for numeric only
        if not match:
            continue
        pair_idx = int(match.group(1))
        prot_a = prot_b = match.group(2) #homodimers have only one unique protein
        key = (prot_a, prot_b)
        if key not in candidates or pair_idx < candidates[key][0]:
            candidates[key] = (pair_idx, d)

    lookup = {}
    for key, (_, d) in candidates.items():
        cif_files = list(d.glob("pred.model_idx_*.cif"))
        npz_files = list(d.glob("scores.model_idx_*.npz"))
        if len(cif_files) != 5 or len(npz_files) != 5:
            lookup[key] = (d, "incomplete_files")
        else:
            lookup[key] = (d, "valid")

    n_valid = sum(1 for _, s in lookup.values() if s == "valid")
    logger.info(f"Found {len(lookup)} pair directories ({n_valid} valid)")
    return lookup


def order_models_by_idx(filename):
    """Extract model index from filename."""
    match = re.search(r'model_idx_(\d+)', str(filename))
    return int(match.group(1)) if match else None


def extract_coords_from_cif(cif_path): #updated extract_coords_from_cif to return chain_residues for check chain order of cif file with chain sequence lengths to verify identity
    """Extract CA coordinates from CIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    coords_dict = {}
    chain_residues = {} #

    for model in structure:
        for chain in model:
            chain_residues[chain.id] = []
            for residue in chain:
                if "CA" in residue:
                    res_num = residue.id[1]
                    ca_coord = residue["CA"].coord
                    coords_dict[(chain.id, res_num)] = ca_coord
                    chain_residues[chain.id].append((res_num, residue.resname))
        break

    return coords_dict, chain_residues


def calculate_distance(coords_dict, chain1, res1, chain2, res2):
    """Calculate CA-CA distance between two residues."""
    coord1 = coords_dict.get((chain1, res1))
    coord2 = coords_dict.get((chain2, res2))

    if coord1 is None or coord2 is None:
        return None

    return float(np.sqrt(np.sum((coord1 - coord2) ** 2)))


def calculate_ipae_with_contacts(pae, cif_file, distance_cutoff=8.0):
    """
    Mean inter-chain PAE at CA–CA contact pairs (same logic as
    SL_collab/processing_chai_outputs_20251015.py).

    Uses mmCIF chains A/B and the same residue ordering as the PAE matrix from
    Chai (chain A block then chain B). That matches native NPZ layout, so no
    CSV protein-order swap is applied here.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('complex', cif_file)
    model = structure[0]

    chain_A = model['A']
    chain_B = model['B']

    A_coords = []
    B_coords = []

    for residue in chain_A:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            A_coords.append(residue['CA'].get_coord())

    for residue in chain_B:
        if residue.get_id()[0] == ' ' and 'CA' in residue:
            B_coords.append(residue['CA'].get_coord())

    A_coords = np.array(A_coords)
    B_coords = np.array(B_coords)

    distance_A_B = np.linalg.norm(
        A_coords[:, np.newaxis] - B_coords[np.newaxis, :], axis=2
    )

    A_B_contact = distance_A_B < distance_cutoff
    A_B_contact_ind = np.array(np.where(A_B_contact))

    A_B_contact_ind_shifted = np.copy(A_B_contact_ind)
    A_B_contact_ind_shifted[1] = A_B_contact_ind_shifted[1] + A_coords.shape[0]

    ipae_contacts = pae[A_B_contact_ind_shifted[0], A_B_contact_ind_shifted[1]]
    mean_ipae_contacts = float(np.mean(ipae_contacts)) if len(ipae_contacts) > 0 else None

    return mean_ipae_contacts


def compute_ipae(pae_matrix, len_protein1):
    """
    Compute interface PAE from the full PAE matrix by averaging both off-diagonal
    blocks. len_protein1 must be the effective split index (already corrected for
    chain swap if needed), so that row/col [:L1] always corresponds to protein A.

    Returns
    -------
    ipae_min_pair : float
        Minimum PAE over all entries in the two interface blocks.
    ipae_1on2, ipae_2on1 : float
        Mean of each interface block (protein-order convention from caller).
    ipae_mean : float
        Mean of ipae_1on2 and ipae_2on1 (overall mean interface PAE).
    """
    pae_matrix = np.array(pae_matrix)
    L1 = int(len_protein1)
    block_1on2 = pae_matrix[:L1, L1:]   # protein A positioned, protein B reference
    block_2on1 = pae_matrix[L1:, :L1]   # protein B positioned, protein A reference
    ipae_1on2 = float(block_1on2.mean())
    ipae_2on1 = float(block_2on1.mean())
    ipae_mean = (ipae_1on2 + ipae_2on1) / 2
    ipae_min_pair = float(np.concatenate([block_1on2.flatten(), block_2on1.flatten()]).min())
    return ipae_min_pair, ipae_1on2, ipae_2on1, ipae_mean

def compute_ipae_at_crosslink(pae_matrix, xl_pos_a, xl_pos_b, len_protein1):
    """
    Extract PAE at the two specific crosslinked residue positions only.

    pae[posA-1, L1+posB-1] — how well posA is predicted given posB as reference (1on2)
    pae[L1+posB-1, posA-1] — how well posB is predicted given posA as reference (2on1)

    Positions are 1-indexed (from CSV), converted to 0-indexed here.
    len_protein1 must be the effective split index (already corrected for swap).

    Returns (ipae_xl, ipae_xl_1on2, ipae_xl_2on1) or (None, None, None) if out of bounds.
    """
    pae_matrix = np.array(pae_matrix)
    L1 = int(len_protein1)
    L_total = pae_matrix.shape[0]
    L2 = L_total - L1

    idx_a = int(xl_pos_a) - 1  # convert to 0-indexed
    idx_b = int(xl_pos_b) - 1  # convert to 0-indexed

    if not (0 <= idx_a < L1):
        logger.warning(f"xl_pos_a={xl_pos_a} out of bounds for protein1 length={L1}")
        return None, None, None
    if not (0 <= idx_b < L2):
        logger.warning(f"xl_pos_b={xl_pos_b} out of bounds for protein2 length={L2}")
        return None, None, None

    ipae_xl_1on2 = float(pae_matrix[idx_a, L1 + idx_b])  # posA predicted, posB as reference
    ipae_xl_2on1 = float(pae_matrix[L1 + idx_b, idx_a])  # posB predicted, posA as reference
    ipae_xl = (ipae_xl_1on2 + ipae_xl_2on1) / 2

    return ipae_xl, ipae_xl_1on2, ipae_xl_2on1


def compute_ipae_at_crosslink_window(
    pae_matrix, xl_pos_a, xl_pos_b, len_protein1, half_window: int = WINDOW_SIZE
):
    """
    Extract mean and median iPAE over a window of residues surrounding the
    inter-chain crosslink site, mirroring extract_crosslink_pae_metrics() in the
    monomer pipeline.

    Because the two residues belong to different proteins (inter-chain), both
    orientations of the window sub-block are purely off-diagonal in the full PAE
    matrix — no diagonal filtering is needed (unlike the monomer case).

    Window construction
    -------------------
    Let rows_a = [max(0, idx_a − hw) .. min(L1−1, idx_a + hw)] (protein-A indices,
    0-based within the PAE matrix rows/cols for protein A).
    Let rows_b = [max(0, idx_b − hw) .. min(L2−1, idx_b + hw)] (protein-B indices,
    0-based within protein B's block).

    Two sub-blocks are collected and concatenated, mirroring the monomer meshgrid
    approach:
        block_1on2 : PAE[rows_a, L1 + rows_b]   — A positioned, B reference
        block_2on1 : PAE[L1 + rows_b, rows_a]   — B positioned, A reference

    All values from both blocks are pooled; mean and median are reported, plus the
    directional means (1on2, 2on1) for transparency.

    Parameters
    ----------
    pae_matrix   : np.ndarray — full (L1+L2) × (L1+L2) PAE matrix
    xl_pos_a     : int — 1-indexed residue on protein A (after swap correction)
    xl_pos_b     : int — 1-indexed residue on protein B (after swap correction)
    len_protein1 : int — effective split index (already corrected for chain swap)
    half_window  : int — residues either side of the crosslink (default WINDOW_SIZE)

    Returns
    -------
    (ipae_xl_window_mean, ipae_xl_window_median, ipae_xl_window_1on2, ipae_xl_window_2on1)
        or (None, None, None, None) if the crosslink position is out of bounds.
    """
    pae_matrix = np.array(pae_matrix)
    L1 = int(len_protein1)
    L_total = pae_matrix.shape[0]
    L2 = L_total - L1

    idx_a = int(xl_pos_a) - 1   # 0-indexed within protein-A block
    idx_b = int(xl_pos_b) - 1   # 0-indexed within protein-B block

    if not (0 <= idx_a < L1):
        logger.warning(
            f"compute_ipae_at_crosslink_window: xl_pos_a={xl_pos_a} out of bounds "
            f"for protein1 length={L1}"
        )
        return None, None, None, None
    if not (0 <= idx_b < L2):
        logger.warning(
            f"compute_ipae_at_crosslink_window: xl_pos_b={xl_pos_b} out of bounds "
            f"for protein2 length={L2}"
        )
        return None, None, None, None

    # Clamp window indices to valid residues within each protein's block
    rows_a = np.arange(max(0, idx_a - half_window), min(L1, idx_a + half_window + 1))
    rows_b = np.arange(max(0, idx_b - half_window), min(L2, idx_b + half_window + 1))

    # Build meshgrids for both orientations (mirrors monomer meshgrid approach)
    r_ab, c_ab = np.meshgrid(rows_a, rows_b, indexing="ij")   # A-rows × B-cols (in A-space)
    r_ba, c_ba = np.meshgrid(rows_b, rows_a, indexing="ij")   # B-rows × A-cols (in B-space)

    # Collect values: protein-B columns are offset by L1 in the full PAE matrix
    vals_1on2 = pae_matrix[r_ab.ravel(), L1 + c_ab.ravel()]   # A positioned, B reference
    vals_2on1 = pae_matrix[L1 + r_ba.ravel(), c_ba.ravel()]   # B positioned, A reference

    if vals_1on2.size == 0 or vals_2on1.size == 0:
        logger.warning(
            f"compute_ipae_at_crosslink_window: empty window for "
            f"xl_pos_a={xl_pos_a}, xl_pos_b={xl_pos_b}"
        )
        return None, None, None, None

    ipae_xl_window_1on2 = float(vals_1on2.mean())
    ipae_xl_window_2on1 = float(vals_2on1.mean())
    all_vals = np.concatenate([vals_1on2, vals_2on1])
    ipae_xl_window_mean = float(all_vals.mean())
    ipae_xl_window_median = float(np.median(all_vals))

    return ipae_xl_window_mean, ipae_xl_window_median, ipae_xl_window_1on2, ipae_xl_window_2on1


def compute_iptm_from_chain_pair(per_chain_pair_iptm, swapped):
    """
    Extract symmetric interface iPTM from per_chain_pair_iptm matrix.
    Shape is (1, 2, 2) — indices follow CIF chain order.

    If chains are swapped relative to the CSV (protein B is chain 0 in the matrix),
    the 1on2 and 2on1 directional values are relabelled so they always refer to
    protein A and protein B as defined in the CSV.

    NOTE: per_chain_pair_iptm chain order is assumed to follow CIF chain order.
    Verify this on a known-swapped case after first run by checking that
    iptm_chai_default matches mat[1,0] in the unswapped case.
    """
    mat = per_chain_pair_iptm[0]  # shape (2, 2)
    # Raw directional values in CIF chain order
    raw_1on2 = float(mat[0, 1])
    raw_2on1 = float(mat[1, 0])
    iptm_mean = (raw_1on2 + raw_2on1) / 2

    # Relabel so 1on2 always means "protein A positioned, protein B reference"
    if swapped:
        iptm_1on2 = raw_2on1  # CIF chain 1 (protein A) positioned, chain 0 (protein B) reference
        iptm_2on1 = raw_1on2
    else:
        iptm_1on2 = raw_1on2
        iptm_2on1 = raw_2on1

    return iptm_mean, iptm_1on2, iptm_2on1


def calculate_chain_pae(pae_matrix, len_protein1):
    """
    Calculate mean intra-chain PAE for each chain, excluding the diagonal.
    len_protein1 must be the effective split index (already corrected for chain
    swap if needed), so [:L1] always corresponds to protein A's block.
    """
    L1 = int(len_protein1)
    chain_A_pae = pae_matrix[:L1, :L1]
    chain_B_pae = pae_matrix[L1:, L1:]

    mask_A = ~np.eye(chain_A_pae.shape[0], dtype=bool)
    mask_B = ~np.eye(chain_B_pae.shape[0], dtype=bool)

    mean_A = float(chain_A_pae[mask_A].mean()) if mask_A.any() else None
    mean_B = float(chain_B_pae[mask_B].mean()) if mask_B.any() else None

    return mean_A, mean_B


def process_ppi_directory(ppi_dir, xl_pos_a, xl_pos_b, len_protein1, len_protein2):
    """
    Process all models in a PPI directory and extract metrics.

    Parameters
    ----------
    ppi_dir : Path
    xl_pos_a : int  — crosslink residue position on protein A (CSV coordinates)
    xl_pos_b : int  — crosslink residue position on protein B (CSV coordinates)
    len_protein1 : int — number of residues in protein A (from proteome)
    """
    # Per-row outputs: no 1on2/2on1 columns in CSV (computed internally only).
    results = {
        'distance_mean': None,
        'distance_std': None,
        'distance_min': None,
        'distance_max': None,
        'intra_pae_protein_a_mean': None,       # mean off-diagonal intra-chain PAE for protein A, averaged across models
        'intra_pae_protein_b_mean': None,       # mean off-diagonal intra-chain PAE for protein B, averaged across models
        'ipae_interface_min_mean_across_models': None,  # mean across models of the per-model minimum value in both interface PAE blocks
        'iptm_mean': None,
        'iptm_chai_default_mean': None,
        'ipae_interface_mean_best_model': None,         # lowest per-model mean interface PAE seen across the 5 models
        'ipae_interface_mean_best_model_idx': None,     # which model achieved ipae_interface_mean_best_model
        'ipae_contact_mean_best_model': None,           # lowest per-model mean iPAE at CA-CA contact pairs (<8 Å)
        'ipae_contact_mean_best_model_idx': None,
        'ipae_contact_mean_best_model_distance': None,  # XL CA-CA distance in that model
        'ipae_xl_site_best_model': None,                # lowest per-model symmetrised iPAE at the exact crosslink residue pair
        'ipae_xl_site_best_model_idx': None,
        'ipae_xl_site_best_model_distance': None,       # XL CA-CA distance in that model
        'ipae_xl_window_mean_best_model': None,         # lowest per-model mean iPAE over ±WINDOW_SIZE residue window around XL site
        'ipae_xl_window_mean_best_model_idx': None,
        'ipae_xl_window_mean_best_model_distance': None,
        'ipae_xl_window_median_best_model': None,       # lowest per-model median iPAE over ±WINDOW_SIZE residue window around XL site
        'ipae_xl_window_median_best_model_idx': None,
        'ipae_xl_window_median_best_model_distance': None,
    }

    for i in range(5):
        results[f'model_{i}_distance'] = None
        results[f'model_{i}_chains_swapped'] = None
        results[f'model_{i}_ipae_interface_min'] = None    # minimum PAE value across both interface blocks
        results[f'model_{i}_ipae_interface_mean'] = None   # mean PAE across both interface blocks
        results[f'model_{i}_ipae_contact_mean'] = None     # mean iPAE at CA-CA contact pairs (<8 Å)
        results[f'model_{i}_iptm'] = None
        results[f'model_{i}_iptm_chai_default'] = None
        results[f'model_{i}_ipae_xl_site'] = None          # symmetrised iPAE at the exact crosslink residue pair
        results[f'model_{i}_ipae_xl_window_mean'] = None   # mean iPAE over ±WINDOW_SIZE residue window around XL site
        results[f'model_{i}_ipae_xl_window_median'] = None # median iPAE over ±WINDOW_SIZE residue window around XL site

    cif_files = list(ppi_dir.glob("pred.model_idx_*.cif"))
    npz_files = list(ppi_dir.glob("scores.model_idx_*.npz"))

    cif_dict = {order_models_by_idx(f): f for f in cif_files if order_models_by_idx(f) is not None}
    npz_dict = {order_models_by_idx(f): f for f in npz_files if order_models_by_idx(f) is not None}

    shared_idx = sorted(set(cif_dict.keys()) & set(npz_dict.keys()))

    distances = []
    chain_A_paes = []
    chain_B_paes = []
    ipaes_min = []
    ipaes_mean = []
    iptms = []
    iptms_chai = []

    for idx in shared_idx:
        cif_file = cif_dict[idx]
        npz_file = npz_dict[idx]

        try:
            # --- Detect chain order from CIF ---
            coords, chain_residues = extract_coords_from_cif(cif_file)
            swapped = False #homodimers have only one unique protein, so no chain swapping needed
            chain_a_id, chain_b_id = "A", "B"
            results[f'model_{idx}_chains_swapped'] = swapped

            if swapped:
                logger.debug(f"  {ppi_dir.name} model {idx}: chains swapped in CIF")

            # --- Crosslink distance.
            #     chain_a_id/chain_b_id point to the correct physical chains.

            distance = calculate_distance(coords, chain_a_id, xl_pos_a, chain_b_id, xl_pos_b)
            if distance is not None:
                results[f'model_{idx}_distance'] = distance
                distances.append(distance)

            # --- Load scores ---
            npz_data = np.load(npz_file, allow_pickle=False)
            pae = npz_data['pae']

            # --- effective_L1 for PAE matrix splitting ---
            effective_L1 = (pae.shape[0] - len_protein1) if swapped else len_protein1

            # --- iPAE: iPAE_mean is mean of the two interface block means (symmetric; no swap needed) ---
            ipae_min_pair, _ipae_1on2, _ipae_2on1, ipae_mean = compute_ipae(pae, len_protein1=effective_L1)

            results[f'model_{idx}_ipae_interface_min'] = ipae_min_pair
            results[f'model_{idx}_ipae_interface_mean'] = ipae_mean
            ipaes_min.append(ipae_min_pair)
            ipaes_mean.append(ipae_mean)

            # --- Mean contact iPAE (native CIF A/B vs PAE layout; see calculate_ipae_with_contacts) ---
            mean_contact_ipae = calculate_ipae_with_contacts(pae, cif_file)
            results[f'model_{idx}_ipae_contact_mean'] = mean_contact_ipae

            # --- iPAE at crosslink ---
            if swapped:
                xl_pos_a_eff, xl_pos_b_eff = xl_pos_b, xl_pos_a
            else:
                xl_pos_a_eff, xl_pos_b_eff = xl_pos_a, xl_pos_b

            ipae_xl, ipae_xl_1on2, ipae_xl_2on1 = compute_ipae_at_crosslink(
                pae,
                xl_pos_a_eff,
                xl_pos_b_eff,
                len_protein1=effective_L1
            )

            results[f'model_{idx}_ipae_xl_site'] = ipae_xl

            # --- iPAE at crosslink window (±WINDOW_SIZE residues around crosslink site) ---
            ipae_xl_window_mean, ipae_xl_window_median, _ipae_xl_window_1on2, _ipae_xl_window_2on1 = (
                compute_ipae_at_crosslink_window(
                    pae,
                    xl_pos_a_eff,
                    xl_pos_b_eff,
                    len_protein1=effective_L1,
                )
            )
            results[f'model_{idx}_ipae_xl_window_mean'] = ipae_xl_window_mean
            results[f'model_{idx}_ipae_xl_window_median'] = ipae_xl_window_median

            # --- iPTM ---
            iptm_mean, iptm_1on2, iptm_2on1 = compute_iptm_from_chain_pair(
                npz_data['per_chain_pair_iptm'], swapped=swapped
            )
            results[f'model_{idx}_iptm'] = iptm_mean
            iptms.append(iptm_mean)

            # --- Chai default iptm (kept for reference) ---
            iptm_chai = float(npz_data['iptm'].item()) if hasattr(npz_data['iptm'], 'item') else float(npz_data['iptm'])
            results[f'model_{idx}_iptm_chai_default'] = iptm_chai
            iptms_chai.append(iptm_chai)

            # --- Intra-chain PAE ---
            block1_pae, block2_pae = calculate_chain_pae(pae, len_protein1=effective_L1)

            if swapped:
                protein_A_pae = block2_pae
                protein_B_pae = block1_pae
            else:
                protein_A_pae = block1_pae
                protein_B_pae = block2_pae

            if protein_A_pae is not None:
                chain_A_paes.append(protein_A_pae)
            if protein_B_pae is not None:
                chain_B_paes.append(protein_B_pae)

        except Exception as e:
            logger.warning(f"Error processing model {idx} in {ppi_dir.name}: {e}")
            continue

    # --- Aggregate across models ---
    if distances:
        results['distance_mean'] = float(np.mean(distances))
        results['distance_std'] = float(np.std(distances, ddof=1)) if len(distances) > 1 else None
        results['distance_min'] = float(np.min(distances))
        results['distance_max'] = float(np.max(distances))
    if chain_A_paes:
        results['intra_pae_protein_a_mean'] = float(np.mean(chain_A_paes))
    if chain_B_paes:
        results['intra_pae_protein_b_mean'] = float(np.mean(chain_B_paes))
    if ipaes_min:
        results['ipae_interface_min_mean_across_models'] = float(np.mean(ipaes_min))
    if iptms:
        results['iptm_mean'] = float(np.mean(iptms))
        results['iptm_chai_default_mean'] = float(np.mean(iptms_chai))

    # --- Best model by mean interface iPAE ---
    mean_candidates = []
    for i in range(5):
        v = results.get(f"model_{i}_ipae_interface_mean")
        if v is not None and not pd.isna(v):
            mean_candidates.append((i, float(v)))

    if mean_candidates:
        min_mean_idx, min_ipae_mean = min(mean_candidates, key=lambda x: x[1])
        results["ipae_interface_mean_best_model"] = min_ipae_mean
        results["ipae_interface_mean_best_model_idx"] = min_mean_idx

    # --- Best model by mean contact iPAE ---
    contact_candidates = []
    for i in range(5):
        v = results.get(f"model_{i}_ipae_contact_mean")
        d = results.get(f"model_{i}_distance")
        if v is not None and not pd.isna(v):
            contact_candidates.append((i, float(v), d))

    if contact_candidates:
        best = min(contact_candidates, key=lambda x: x[1])
        results["ipae_contact_mean_best_model"] = best[1]
        results["ipae_contact_mean_best_model_idx"] = best[0]
        results["ipae_contact_mean_best_model_distance"] = (
            float(best[2]) if best[2] is not None and not pd.isna(best[2]) else None
        )

    # --- Best model by iPAE at exact crosslink site ---
    xl_candidates = []
    for i in range(5):
        v = results.get(f"model_{i}_ipae_xl_site")
        d = results.get(f"model_{i}_distance")
        if v is not None and not pd.isna(v):
            xl_candidates.append((i, float(v), d))

    if xl_candidates:
        min_model_idx, min_ipae_xl, xl_dist = min(xl_candidates, key=lambda x: x[1])
        results["ipae_xl_site_best_model"] = min_ipae_xl
        results["ipae_xl_site_best_model_idx"] = min_model_idx
        results["ipae_xl_site_best_model_distance"] = (
            float(xl_dist) if xl_dist is not None and not pd.isna(xl_dist) else None
        )

    # --- Best model by mean iPAE over crosslink window ---
    xl_window_candidates = []
    for i in range(5):
        v = results.get(f"model_{i}_ipae_xl_window_mean")
        d = results.get(f"model_{i}_distance")
        if v is not None and not pd.isna(v):
            xl_window_candidates.append((i, float(v), d))

    if xl_window_candidates:
        best_window = min(xl_window_candidates, key=lambda x: x[1])
        results["ipae_xl_window_mean_best_model"] = best_window[1]
        results["ipae_xl_window_mean_best_model_idx"] = best_window[0]
        results["ipae_xl_window_mean_best_model_distance"] = (
            float(best_window[2])
            if best_window[2] is not None and not pd.isna(best_window[2])
            else None
        )

    # --- Best model by median iPAE over crosslink window ---
    xl_window_median_candidates = []
    for i in range(5):
        v = results.get(f"model_{i}_ipae_xl_window_median")
        d = results.get(f"model_{i}_distance")
        if v is not None and not pd.isna(v):
            xl_window_median_candidates.append((i, float(v), d))

    if xl_window_median_candidates:
        best_window_med = min(xl_window_median_candidates, key=lambda x: x[1])
        results["ipae_xl_window_median_best_model"] = best_window_med[1]
        results["ipae_xl_window_median_best_model_idx"] = best_window_med[0]
        results["ipae_xl_window_median_best_model_distance"] = (
            float(best_window_med[2])
            if best_window_med[2] is not None and not pd.isna(best_window_med[2])
            else None
        )

    return results


def save_checkpoint(valid_rows, failed_rows, results_dir):
    """Save checkpoint files."""
    checkpoint_output = results_dir / "checkpoint_naive_chai_processing_20260417.csv"
    checkpoint_failed = results_dir / "checkpoint_failed_naive_chai_processing_20260417.csv"

    if valid_rows:
        pd.DataFrame(valid_rows).to_csv(checkpoint_output, index=False)
        logger.info(f"Checkpoint: Saved {len(valid_rows)} valid PPIs")

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(checkpoint_failed, index=False)
        logger.info(f"Checkpoint: Saved {len(failed_rows)} failed PPIs")


def main():
    """Main processing pipeline."""
    logger.info("Starting PPI crosslink analysis pipeline (naive chai processing)")

    df_ppi, df_proteome, df_xlms = load_data()
    proteome_mapping = create_proteome_mapping(df_proteome)
    pair_dir_lookup = build_pair_dir_lookup(CHAI_HOMODIMER_DIR))
    df_ppi_homodimers = build_homodimer_ppi_self_pairs(df_ppi)
    df_expanded = expand_ppi_with_crosslinks(df_ppi_homodimers, df_xlms)

    # Temporary diagnostic — remove after checking
    problem_rows = df_expanded[
        df_expanded.apply(lambda r: (
            pd.notna(r.get('Crosslink Position A')) and
            pd.notna(r.get('Protein A')) and
            proteome_mapping.get(r['Protein A']) is not None and
            int(float(r['Crosslink Position A'])) > len(proteome_mapping[r['Protein A']])
        ), axis=1)
    ]
    if not problem_rows.empty:
        logger.warning(f"Rows where xl_pos_a exceeds protein A length:\n"
                    f"{problem_rows[['Protein A','Protein B','Crosslink Position A','Crosslink Position B']].to_string()}")

    valid_rows = []
    failed_rows = []
    processed_count = 0
    save_interval = 50

    for idx, row in df_expanded.iterrows():
        protein_a = row['Protein A']
        protein_b = row['Protein B']

        logger.info(f"Processing {idx + 1}/{len(df_expanded)}: {protein_a} - {protein_b}")

        dir_entry = pair_dir_lookup.get((protein_a, protein_b))
        if dir_entry is None:
            logger.warning(f"Skipping {protein_a} - {protein_b}: no_directory_found")
            failed_row = row.to_dict()
            failed_row['failure_reason'] = "no_directory_found"
            failed_rows.append(failed_row)
            continue

        ppi_dir, status = dir_entry
        if status != "valid":
            logger.warning(f"Skipping {protein_a} - {protein_b}: {status}")
            failed_row = row.to_dict()
            failed_row['failure_reason'] = status
            failed_rows.append(failed_row)
            continue

        xl_pos_a = row.get('Crosslink Position A')
        xl_pos_b = row.get('Crosslink Position B')

        if pd.isna(xl_pos_a) or pd.isna(xl_pos_b):
            logger.warning(f"Missing crosslink positions for {protein_a} - {protein_b}")
            failed_row = row.to_dict()
            failed_row['failure_reason'] = "missing_crosslink_positions"
            failed_rows.append(failed_row)
            continue

        seq_a = proteome_mapping.get(protein_a)
        if seq_a is None:
            logger.warning(f"No sequence found for {protein_a}, skipping")
            failed_row = row.to_dict()
            failed_row['failure_reason'] = "missing_sequence"
            failed_rows.append(failed_row)
            continue
        len_protein1 = len(seq_a)

        seq_b = proteome_mapping.get(protein_b)
        if seq_b is None:
            logger.warning(f"No sequence found for {protein_b}, skipping")
            failed_row = row.to_dict()
            failed_row["failure_reason"] = "missing_sequence"
            failed_rows.append(failed_row)
            continue
        len_protein2 = len(seq_b)

        # Sanity check: crosslink positions must be within sequence bounds
        if int(xl_pos_a) > len_protein1:
            logger.warning(
                f"xl_pos_a={int(xl_pos_a)} exceeds {protein_a} length={len_protein1} — "
                f"check if crosslink assignment is flipped in XL-MS data"
            )
        if int(xl_pos_b) > len_protein2:
            logger.warning(
                f"xl_pos_b={int(xl_pos_b)} exceeds {protein_b} length={len_protein2} — "
                f"check if crosslink assignment is flipped in XL-MS data"
    )

        try:
            results = process_ppi_directory(ppi_dir, int(xl_pos_a), int(xl_pos_b), len_protein1=len_protein1, len_protein2=len_protein2)

            output_row = row.to_dict()
            output_row.update(results)
            valid_rows.append(output_row)

            processed_count += 1

            if processed_count % save_interval == 0:
                logger.info(f"Saving checkpoint at {processed_count} processed PPIs...")
                save_checkpoint(valid_rows, failed_rows, RESULTS_DIR)

        except Exception as e:
            logger.error(f"Error processing {protein_a} - {protein_b}: {e}")
            failed_row = row.to_dict()
            failed_row['failure_reason'] = f"processing_error: {str(e)}"
            failed_rows.append(failed_row)

    if valid_rows:
        df_output = pd.DataFrame(valid_rows)
        df_output.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Final: Saved {len(valid_rows)} valid PPIs to {OUTPUT_CSV}")

    if failed_rows:
        df_failed = pd.DataFrame(failed_rows)
        df_failed.to_csv(FAILED_CSV, index=False)
        logger.info(f"Final: Saved {len(failed_rows)} failed PPIs to {FAILED_CSV}")

    logger.info(f"Pipeline complete! Total processed: {processed_count} PPIs")

if __name__ == "__main__":
    main()