# StarPath: Toxoplasma gondii PPI Network Structural Analysis

All predictions available for interactive preview here: https://starpath.wi.mit.edu/

StarPath provides a network of 2,859 protein-protein interactions (PPIs), resolved from 29,624 recorded crosslinks. This repo integrates the empirical crosslinking mass-spectrometry (XL-MS) data with Chai-1 structure prediction and analysis. 

Data: 
* xlms_data: contains all empirical PPI and crosslink-secific XL-MS data and T. gondii RH88 proteome
* processed_chai_outputs: contains all analysis outputs from '\*\_process\_\*.py' scripts. Includes PAE, iPAE, PTM, iPTM, and crosslink-site distance metrics

Scripts: 
* subsetted by prediction type. Includes all scripts needed for inference and processing the inference outputs.
* NOTE: By default, Chai prediction outputs are written to data/chai_outputs/ inside the repository. To store them elsewhere (e.g. on a separate disk due to file size), set the TOXO_CHAI_OUTPUTS_DIR environment variable before running: export TOXO_CHAI_OUTPUTS_DIR=/path/to/your/output/directory

Notebooks: 
* Contains all analysis scripts for generating structurally-relevant figures


Reference: Butterworth, S., Gin, A., Shikha, S., Tengganu U., Rush, J., Duraisingh, T., Sodeinde, V., Lembgruber, L., Schulte, F., Hu, K., Sheiner, L., Ovchinnikov, S., & Lourido, S. Proteome-wide crosslinking mass spectrometry reveals novel components of essential complexes in Toxoplasma. [unpublished]. (2026).

Author: Ashley L. Gin
