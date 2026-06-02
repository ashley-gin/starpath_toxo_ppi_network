"""
This file contains the configuration for the toxo ppi network project.
It defines the file paths for the input and output data -- NOTE: all paths are relative to the repo root.
outputs to CHAI_OUTPUTS_DIR are large, and are added to .gitignore.
"""

import os
from pathlib import Path

#absolute path to the repo root -- for user compatibility
REPO_ROOT = Path(__file__).resolve().parent

#--- Input Toxoplasma reference data --
XLMS_DATA_DIR = REPO_ROOT / "data" / "xlms_data"

#--- Specific Reference files ---
CROSSLINKS_DIR    = XLMS_DATA_DIR / "rh88_crosslinks"
PROTEOME_DIR      = XLMS_DATA_DIR / "rh88_proteome"
PROTEOME_CSV         = PROTEOME_DIR / "toxodb_tgrh88_protein_sequences.csv"
INTER_XL_CSV         = CROSSLINKS_DIR / "inter_crosslinks_unique.csv"
NULL_INTER_XL_CSV    = CROSSLINKS_DIR / "null_inter_crosslinks_random_lys.csv"
INTRA_XL_CSV         = CROSSLINKS_DIR / "intra_crosslinks_unique.csv"
NULL_INTRA_XL_CSV    = CROSSLINKS_DIR / "null_intra_crosslinks_random_lys.csv"
PPI_CSV     = CROSSLINKS_DIR / "combined_data_ppis.csv"

#--- Processed chai prediction outputs (.csv files -- for figure recreation) ---
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed_chai_outputs"

#--- Raw Chai prediction outputs (added to .gitignore for size) ---
CHAI_OUTPUTS_DIR = Path(os.environ.get("TOXO_CHAI_OUTPUTS_DIR", str(REPO_ROOT / "data" / "chai_outputs")))
CHAI_MONOMER_DIR = CHAI_OUTPUTS_DIR / "naive_chai_monomers"
CHAI_HOMODIMER_DIR = CHAI_OUTPUTS_DIR / "naive_chai_homodimers"
CHAI_MULTIMER_DIR = CHAI_OUTPUTS_DIR / "chai_multimers"