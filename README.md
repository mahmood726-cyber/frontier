# frontier: 2026 Frontier Statistical Methods for Evidence Synthesis

This repository implements the most advanced "journal-grade" (2025/2026) statistical methodologies for synthesizing Cardiovascular Disease (CVD) evidence across heterogeneous global datasets.

## Frontier Methods
1.  **Conformalized Meta-Learning (CML):** Provides distribution-free, finite-sample coverage guarantees for treatment effects by reweighting noisy IHME data against a "clean" CT.gov anchor set (Liu et al., ICLR 2026).
2.  **Wasserstein-over-Wasserstein (WoW) Distance:** A geometric alignment metric that quantifies the distance between entire study populations (e.g., comparing IHME burden distributions vs. RCT population characteristics).
3.  **Stein Variational Gradient Descent (SVGD):** A particle-based Bayesian inference method that captures complex, non-Gaussian posterior distributions more accurately than standard MCMC.

## Project Scope
- **Data Integration:** Synthesizes gold-standard **ClinicalTrials.gov** RCTs with high-dimensional **IHME** burden data and **World Bank** economic covariates.
- **Goal:** Robust, distribution-free uncertainty quantification and geometric alignment of heterogeneous global health datasets.
- **E156 Deliverables:** 7-sentence micro-paper with **TruthCert** proof-carrying numbers.

## Structure
- `src/`: Python scripts for CML, WoW, and SVGD implementation.
- `data/`: Ingested (Open Access) data and distribution fixtures.
- `output/`: Results, prediction sets, and TruthCert audit logs.
- `tests/`: Automated test suite for statistical consistency and coverage.
- `docs/`: E156 micro-paper and methodology documentation.

## Deployment
Interactive dashboard hosted at `mahmood726-cyber.github.io/frontier/`.
