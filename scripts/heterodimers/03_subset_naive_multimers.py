#!/usr/bin/env python
# coding: utf-8

"""
Subset naive multimer processing results using XL-site metrics from Chai output:

- Multimer confidence: ``iPAE_XL_min`` (not per-model min interface iPAE).
- Crosslink distance: ``iPAE_XL_min_distance`` (CA–CA at the model that minimizes XL iPAE).

Two output modes (separate subfolders):

1. **by_crosslink** — one row per crosslink (same grain as the input); same step1/step2
   split as the original 05_subsetting_naive_ppis notebook.

2. **by_ppi** — one row per unique (Protein A, Protein B), with PPI-level buckets:
   a PPI is **unconfident** if any crosslink has ``iPAE_XL_min`` above the threshold;
   **out of range** (distance) if any crosslink has ``iPAE_XL_min_distance`` above the
   distance threshold.  Also writes ``ppi_crosslink_detail_*.csv`` with every crosslink
   row plus duplicate PPI-level flags on each row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = "/home/ubuntu/chai-lab/updated_analyses_SL"
RESULTS_DIR = Path(f"{BASE_DIR}/chai_processed_results")
INPUT_CSV = RESULTS_DIR / "naive_chai_multimer_processing_20260329.csv"
ANALYSIS_ROOT = RESULTS_DIR / "subsetted_naive_chai_20260329"
DIR_CROSSLINK = ANALYSIS_ROOT / "by_crosslink"
DIR_PPI = ANALYSIS_ROOT / "by_ppi"

IPAE_CONFIDENT_THRESHOLD = 15.0
DISTANCE_THRESHOLD = 35.0  # Angstroms

CROSSLINK_KEY = [
    "Protein A",
    "Protein B",
    "Crosslink Position A",
    "Crosslink Position B",
    "Crosslinked Residue A",
    "Crosslinked Residue B",
]


def load_processed_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    logger.info("Loaded %s rows from %s", len(df), filepath)
    logger.info("Unique PPIs (Protein A & B): %s", df.groupby(["Protein A", "Protein B"]).ngroups)
    logger.info("Unique crosslinks: %s", df.groupby(CROSSLINK_KEY, dropna=False).ngroups)
    return df


def add_crosslink_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Uses existing iPAE_XL_min and iPAE_XL_min_distance columns."""
    required = ["iPAE_XL_min", "iPAE_XL_min_distance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    out = df.copy()

    ipae = pd.to_numeric(out["iPAE_XL_min"], errors="coerce")
    dist = pd.to_numeric(out["iPAE_XL_min_distance"], errors="coerce")

    # Missing iPAE → not confident; missing distance → treat as out of range
    out["crosslink_multimer_confident"] = ipae.notna() & (ipae <= IPAE_CONFIDENT_THRESHOLD)
    out["crosslink_distance_in_range"] = dist.notna() & (dist <= DISTANCE_THRESHOLD)

    return out


def analyze_multimer_confidence_crosslink(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    logger.info("\n%s", "=" * 80)
    logger.info(
        "STEP 1 (by crosslink): multimer confidence via iPAE_XL_min ≤ %s",
        IPAE_CONFIDENT_THRESHOLD,
    )
    logger.info("%s", "=" * 80)

    confident = df[df["crosslink_multimer_confident"]].copy()
    unconfident = df[~df["crosslink_multimer_confident"]].copy()

    subsets = {
        "confident_multimer": confident,
        "unconfident_multimer": unconfident,
    }

    for name, sub in subsets.items():
        out = DIR_CROSSLINK / f"step1_{name}.csv"
        sub.to_csv(out, index=False)
        logger.info("Saved %s: %s rows → %s", name, len(sub), out)

    _log_step1_stats_crosslink(df, confident, unconfident, DIR_CROSSLINK / "step1_multimer_statistics.csv")
    return subsets


def analyze_crosslink_distance_crosslink(subsets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("\n%s", "=" * 80)
    logger.info(
        "STEP 2 (by crosslink): distance via iPAE_XL_min_distance ≤ %s Å",
        DISTANCE_THRESHOLD,
    )
    logger.info("%s", "=" * 80)

    distance_subsets: dict[str, pd.DataFrame] = {}

    for parent_name, parent_df in subsets.items():
        logger.info("\nParent: %s (%s rows)", parent_name, len(parent_df))

        in_range = parent_df[parent_df["crosslink_distance_in_range"]].copy()
        out_of_range = parent_df[~parent_df["crosslink_distance_in_range"]].copy()

        ir_name = f"{parent_name}_in_range"
        oor_name = f"{parent_name}_out_of_range"
        distance_subsets[ir_name] = in_range
        distance_subsets[oor_name] = out_of_range

        in_range.to_csv(DIR_CROSSLINK / f"step2_{ir_name}.csv", index=False)
        out_of_range.to_csv(DIR_CROSSLINK / f"step2_{oor_name}.csv", index=False)
        logger.info("  %s: %s rows", ir_name, len(in_range))
        logger.info("  %s: %s rows", oor_name, len(out_of_range))

        _log_step2_stats_crosslink(parent_name, parent_df, in_range, out_of_range)

    _write_step2_stats_crosslink(subsets, distance_subsets, DIR_CROSSLINK / "step2_distance_statistics.csv")
    return distance_subsets


def _ppi_aggregate(g: pd.DataFrame) -> pd.Series:
    """One row per PPI: any-unconfident / any-out-of-range semantics."""
    any_unconfident = (~g["crosslink_multimer_confident"]).any()
    any_oor = (~g["crosslink_distance_in_range"]).any()
    n_xl = len(g)
    n_conf = int(g["crosslink_multimer_confident"].sum())
    n_in_d = int(g["crosslink_distance_in_range"].sum())
    return pd.Series(
        {
            "n_crosslinks": n_xl,
            "n_crosslinks_multimer_confident": n_conf,
            "n_crosslinks_multimer_unconfident": n_xl - n_conf,
            "n_crosslinks_distance_in_range": n_in_d,
            "n_crosslinks_distance_out_of_range": n_xl - n_in_d,
            "ppi_any_crosslink_unconfident": bool(any_unconfident),
            "ppi_all_crosslinks_confident": bool(not any_unconfident),
            "ppi_any_crosslink_distance_out_of_range": bool(any_oor),
            "ppi_all_crosslinks_distance_in_range": bool(not any_oor),
        }
    )


def build_ppi_summary(df: pd.DataFrame) -> pd.DataFrame:
    ppi = (
        df.groupby(["Protein A", "Protein B"], as_index=False)
        .apply(_ppi_aggregate, include_groups=False)
        .reset_index(drop=True)
    )
    # groupby.apply with as_index=False can leave MultiIndex; ensure flat
    if isinstance(ppi.columns, pd.MultiIndex):
        ppi.columns = [c[-1] if isinstance(c, tuple) else c for c in ppi.columns]

    ppi["ppi_multimer_group"] = np.where(
        ppi["ppi_any_crosslink_unconfident"], "unconfident_multimer", "confident_multimer"
    )
    ppi["ppi_distance_group"] = np.where(
        ppi["ppi_any_crosslink_distance_out_of_range"],
        "out_of_range",
        "in_range",
    )
    return ppi


def _ppi_ordered_pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Unique PPIs as stored: (Protein A, Protein B) tuple identity."""
    a = df["Protein A"].astype(str)
    b = df["Protein B"].astype(str)
    return set(zip(a, b, strict=True))


def _ppi_unordered_pair_set(df: pd.DataFrame) -> set[frozenset[str]]:
    """Same two proteins regardless of A/B column assignment."""
    return {
        frozenset((x, y))
        for x, y in zip(df["Protein A"].astype(str), df["Protein B"].astype(str), strict=True)
    }


def ppi_pair_overlap_counts(conf: pd.DataFrame, unconf: pd.DataFrame) -> tuple[int, int]:
    """
    Return (n_overlapping_ordered_pairs, n_overlapping_unordered_pairs) between two PPI tables.
    Ordered overlap uses exact (Protein A, Protein B) matches.
    """
    o_c, o_u = _ppi_ordered_pair_set(conf), _ppi_ordered_pair_set(unconf)
    u_c, u_u = _ppi_unordered_pair_set(conf), _ppi_unordered_pair_set(unconf)
    return len(o_c & o_u), len(u_c & u_u)


def merge_ppi_flags_onto_crosslinks(df: pd.DataFrame, ppi_summary: pd.DataFrame) -> pd.DataFrame:
    meta = ppi_summary[
        [
            "Protein A",
            "Protein B",
            "n_crosslinks",
            "n_crosslinks_multimer_confident",
            "n_crosslinks_multimer_unconfident",
            "n_crosslinks_distance_in_range",
            "n_crosslinks_distance_out_of_range",
            "ppi_any_crosslink_unconfident",
            "ppi_all_crosslinks_confident",
            "ppi_any_crosslink_distance_out_of_range",
            "ppi_all_crosslinks_distance_in_range",
            "ppi_multimer_group",
            "ppi_distance_group",
        ]
    ]
    return df.merge(meta, on=["Protein A", "Protein B"], how="left", validate="m:1")


def save_ppi_mode(ppi_summary: pd.DataFrame, df_with_ppi: pd.DataFrame) -> None:
    logger.info("\n%s", "=" * 80)
    logger.info("PPI mode: one row per (Protein A, Protein B)")
    logger.info("%s", "=" * 80)

    # Step 1 style: confident = no crosslink unconfident; unconfident = any unconfident
    conf = ppi_summary[ppi_summary["ppi_multimer_group"] == "confident_multimer"].copy()
    unconf = ppi_summary[ppi_summary["ppi_multimer_group"] == "unconfident_multimer"].copy()

    n_ov_o, n_ov_u = ppi_pair_overlap_counts(conf, unconf)
    logger.info(
        "PPI overlap (step1 confident vs unconfident): %s ordered (A,B) pairs, %s unordered protein pairs",
        n_ov_o,
        n_ov_u,
    )
    if n_ov_o or n_ov_u:
        o_c, o_u = _ppi_ordered_pair_set(conf), _ppi_ordered_pair_set(unconf)
        shared = sorted(o_c & o_u)[:20]
        logger.error("Overlapping ordered PPI pairs (showing up to 20): %s", shared)
    assert n_ov_o == 0 and n_ov_u == 0, (
        "step1 confident and unconfident PPI outputs must be disjoint on (Protein A, Protein B); "
        f"found {n_ov_o} ordered and {n_ov_u} unordered overlaps"
    )

    conf.to_csv(DIR_PPI / "step1_confident_multimer_ppi.csv", index=False)
    unconf.to_csv(DIR_PPI / "step1_unconfident_multimer_ppi.csv", index=False)
    logger.info("step1 confident PPIs: %s → %s", len(conf), DIR_PPI / "step1_confident_multimer_ppi.csv")
    logger.info("step1 unconfident PPIs: %s → %s", len(unconf), DIR_PPI / "step1_unconfident_multimer_ppi.csv")

    # Step 2: split each multimer group by PPI-level distance bucket
    for multimer_label, sub in [("confident_multimer", conf), ("unconfident_multimer", unconf)]:
        in_r = sub[sub["ppi_distance_group"] == "in_range"].copy()
        oor = sub[sub["ppi_distance_group"] == "out_of_range"].copy()
        in_r.to_csv(DIR_PPI / f"step2_{multimer_label}_in_range_ppi.csv", index=False)
        oor.to_csv(DIR_PPI / f"step2_{multimer_label}_out_of_range_ppi.csv", index=False)
        logger.info(
            "  step2 %s_in_range_ppi: %s PPIs",
            multimer_label,
            len(in_r),
        )
        logger.info(
            "  step2 %s_out_of_range_ppi: %s PPIs",
            multimer_label,
            len(oor),
        )

    # Full crosslink grain + PPI flags (for tracking each XL while knowing PPI bucket)
    detail_path = DIR_PPI / "ppi_crosslink_detail_with_ppi_flags.csv"
    df_with_ppi.to_csv(detail_path, index=False)
    logger.info("Crosslink detail with PPI flags: %s rows → %s", len(df_with_ppi), detail_path)

    stats = [
        {
            "Category": "Total unique PPIs",
            "Count": len(ppi_summary),
        },
        {
            "Category": "PPI confident multimer (all XLs confident)",
            "Count": int((~ppi_summary["ppi_any_crosslink_unconfident"]).sum()),
        },
        {
            "Category": "PPI unconfident multimer (≥1 XL unconfident)",
            "Count": int(ppi_summary["ppi_any_crosslink_unconfident"].sum()),
        },
        {
            "Category": "PPI all XL distances in range",
            "Count": int((~ppi_summary["ppi_any_crosslink_distance_out_of_range"]).sum()),
        },
        {
            "Category": "PPI ≥1 XL distance out of range",
            "Count": int(ppi_summary["ppi_any_crosslink_distance_out_of_range"].sum()),
        },
    ]
    pd.DataFrame(stats).to_csv(DIR_PPI / "ppi_mode_summary_statistics.csv", index=False)


def _log_step1_stats_crosslink(
    df: pd.DataFrame,
    confident: pd.DataFrame,
    unconfident: pd.DataFrame,
    stats_path: Path,
) -> None:
    logger.info("\n%s", "-" * 80)
    logger.info("Step 1 (crosslink): counts by unique crosslink")
    logger.info("%s", "-" * 80)

    def n_xl(x):
        return x.groupby(CROSSLINK_KEY, dropna=False).ngroups

    total_xl = n_xl(df)
    c_xl = n_xl(confident)
    u_xl = n_xl(unconfident)
    logger.info("Total unique crosslinks:     %6d", total_xl)
    logger.info("Confident (iPAE_XL_min≤%s): %6d (%.2f%%)", IPAE_CONFIDENT_THRESHOLD, c_xl, 100 * c_xl / total_xl if total_xl else 0)
    logger.info("Unconfident:                 %6d (%.2f%%)", u_xl, 100 * u_xl / total_xl if total_xl else 0)

    stats_df = pd.DataFrame(
        [
            {"Category": "Total unique crosslinks", "Count": total_xl},
            {
                "Category": f"Confident (iPAE_XL_min ≤ {IPAE_CONFIDENT_THRESHOLD})",
                "Count": c_xl,
            },
            {"Category": "Unconfident", "Count": u_xl},
        ]
    )
    stats_df.to_csv(stats_path, index=False)


def _log_step2_stats_crosslink(
    parent_name: str,
    parent_df: pd.DataFrame,
    in_range: pd.DataFrame,
    out_of_range: pd.DataFrame,
) -> None:
    def n_xl(x):
        return x.groupby(CROSSLINK_KEY, dropna=False).ngroups

    parent_u = n_xl(parent_df)
    ir_u = n_xl(in_range)
    oor_u = n_xl(out_of_range)
    logger.info("  Unique crosslinks in parent: %d", parent_u)
    logger.info(
        "    In range (≤ %s Å):     %6d (%.2f%%)",
        DISTANCE_THRESHOLD,
        ir_u,
        100 * ir_u / parent_u if parent_u else 0,
    )
    logger.info(
        "    Out of range:          %6d (%.2f%%)",
        oor_u,
        100 * oor_u / parent_u if parent_u else 0,
    )


def _write_step2_stats_crosslink(
    multimer_subsets: dict[str, pd.DataFrame],
    distance_subsets: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    rows = []

    def n_xl(x):
        return x.groupby(CROSSLINK_KEY, dropna=False).ngroups

    for name, subset_df in distance_subsets.items():
        if name.endswith("_in_range"):
            parent = name[: -len("_in_range")]
            dist_cat = f"In range (≤ {DISTANCE_THRESHOLD} Å)"
        else:
            parent = name[: -len("_out_of_range")]
            dist_cat = "Out of range"

        parent_unique = n_xl(multimer_subsets[parent])
        xl_count = n_xl(subset_df)
        pct = (xl_count / parent_unique * 100) if parent_unique > 0 else 0.0
        rows.append(
            {
                "Parent Group": parent,
                "Distance Category": dist_cat,
                "Subset Name": name,
                "Unique Crosslinks": xl_count,
                "Percentage of Parent": f"{pct:.2f}%",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_report(
    df: pd.DataFrame,
    multimer_subsets: dict[str, pd.DataFrame],
    distance_subsets: dict[str, pd.DataFrame],
    ppi_summary: pd.DataFrame,
) -> None:
    lines = [
        "=" * 80,
        "SUBSETTING SUMMARY (naive_chai_multimer_processing_20260329)",
        "=" * 80,
        f"Input rows: {len(df)}",
        f"Unique crosslinks: {df.groupby(CROSSLINK_KEY, dropna=False).ngroups}",
        f"Unique PPIs: {df.groupby(['Protein A', 'Protein B']).ngroups}",
        "",
        f"Thresholds: iPAE_XL_min ≤ {IPAE_CONFIDENT_THRESHOLD} (confident); "
        f"iPAE_XL_min_distance ≤ {DISTANCE_THRESHOLD} Å (in range)",
        "",
        "--- by_crosslink (step1 / step2) ---",
    ]
    for n, s in multimer_subsets.items():
        lines.append(f"  {n}: {len(s)} rows")
    for n, s in distance_subsets.items():
        lines.append(f"  {n}: {len(s)} rows")

    lines.extend(
        [
            "",
            "--- by_ppi ---",
            f"  Unique PPIs: {len(ppi_summary)}",
            f"  Confident multimer PPIs (all XLs): {(~ppi_summary['ppi_any_crosslink_unconfident']).sum()}",
            f"  Unconfident multimer PPIs (≥1 XL): {ppi_summary['ppi_any_crosslink_unconfident'].sum()}",
            f"  PPIs with ≥1 XL distance out of range: {ppi_summary['ppi_any_crosslink_distance_out_of_range'].sum()}",
        ]
    )
    conf_r = ppi_summary[ppi_summary["ppi_multimer_group"] == "confident_multimer"]
    unconf_r = ppi_summary[ppi_summary["ppi_multimer_group"] == "unconfident_multimer"]
    ov_o, ov_u = ppi_pair_overlap_counts(conf_r, unconf_r)
    lines.extend(
        [
            f"  Overlap of unique (Protein A, Protein B) pairs (step1 conf vs unconf multimer): {ov_o}",
            f"  Overlap of unique unordered protein pairs (same two IDs, any A/B order): {ov_u}",
            "",
            "=" * 80,
        ]
    )
    report_path = ANALYSIS_ROOT / "subsetting_summary_report_20260329.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    print("\n".join(lines))


def main() -> None:
    logger.info("05test_subsetting_naive_ppis_20260329")
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    DIR_CROSSLINK.mkdir(parents=True, exist_ok=True)
    DIR_PPI.mkdir(parents=True, exist_ok=True)

    df = load_processed_data(INPUT_CSV)
    df = add_crosslink_classifications(df)

    multimer_subsets = analyze_multimer_confidence_crosslink(df)
    distance_subsets = analyze_crosslink_distance_crosslink(multimer_subsets)

    ppi_summary = build_ppi_summary(df)
    df_with_ppi = merge_ppi_flags_onto_crosslinks(df, ppi_summary)
    save_ppi_mode(ppi_summary, df_with_ppi)

    write_summary_report(df, multimer_subsets, distance_subsets, ppi_summary)

    logger.info("\n%s", "=" * 80)
    logger.info("Done. Outputs under %s", ANALYSIS_ROOT)
    logger.info("%s", "=" * 80)


if __name__ == "__main__":
    main()
