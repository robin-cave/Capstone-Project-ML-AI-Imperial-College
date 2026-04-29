# Datasheet: BBO Capstone Query–Response Dataset

This datasheet documents the dataset produced and accumulated during the Imperial College Machine Learning and Artificial Intelligence **Bayesian Black-Box Optimisation (BBO)** capstone. It follows the framework from Mini-lesson 21.1 (motivation, composition, collection, preprocessing, uses, distribution, maintenance).

## Motivation

- **Purpose:** The dataset exists to support **sequential optimisation** of eight synthetic black-box objective functions under a strict evaluation budget: **one query per function per week** across thirteen submission rounds (the twelve scheduled weeks plus a W13 bonus round). It captures every submitted design point and the corresponding oracle response, enabling analysis of exploration–exploitation trade-offs, surrogate-based acquisition, and strategy evolution under extreme data scarcity.
- **Task supported:** Empirical study of **Bayesian optimisation-style** workflows (Gaussian processes, alternative surrogates, acquisition functions, manual and hybrid probes) on problems from 2D to 8D in the unit hypercube.
- **Creator and context:** Created by the student author as coursework for the BBO capstone. The **initial design** and **oracle** (hidden objective functions) are provided by the course; **weekly queries** are chosen by the student using the code and notebooks in this repository.
- **Funding:** Academic coursework only; no external funding for dataset creation.

## Composition

- **What each instance represents:** One **evaluation** of one synthetic function: a vector **x** in **[0, 1]^d** (d = 2, 3, 4, 5, 6, or 8 depending on the function id) and a **scalar output** y returned by the black box.
- **Structure:**
  - **Per function:** `initial_inputs.npy` / `initial_outputs.npy` (course-provided starting designs; sample counts scale with dimensionality, roughly 10–40 points per function).
  - **Weekly batches:** For each week *k*, `public/data/results/week_k/inputs.txt` and `outputs.txt` list the eight submitted points and received values (one row per function per week, in function order).
  - **Optional checkpoints:** `week_k_inputs.npy` / `week_k_outputs.npy` under `data/function_*` store **cumulative** design matrices and response vectors after ingesting week *k* (when the data-management workflow is run).
- **Size (order of magnitude):** After *n* weeks, each function has on the order of **initial samples + n** observations (e.g. ~18 + *n* for 2D functions with a larger initial design, up to ~40 + *n* for 8D, depending on course baseline). At *n* = 13 (the current total) this yields on the order of 30–55 observations per function.
- **Format:** Floating-point arrays (`.npy`); weekly portal logs as text representations of NumPy arrays.
- **Missing data:** None in the stored artefacts; every submitted point has a recorded response.
- **Sensitive or confidential content:** **No.** Synthetic objectives only; no personal data, no privileged records, no communications content.

## Collection process

- **Acquisition of initial data:** Provided by the course as fixed `.npy` files per function.
- **Acquisition of weekly data:** Each week, the student runs `notebooks/weekly_workflow.ipynb` (and related modules) to fit **surrogate models** (e.g. GP with RBF/ARD, SVR bootstrap ensemble, optional MLP) and optimise **acquisition functions** (UCB, EI, PI) with optional **regional focus** (bounding box around a hand-picked anchor). Some weeks include **manual** queries (e.g. boundary probes, grid corners) driven by written strategy reports.
- **Sampling strategy:** **Adaptive sequential design** — not i.i.d. sampling from the domain. Later points cluster near historically strong regions (exploitation) or test explicit hypotheses (e.g. corners, boundaries). This induces **strong spatial and temporal bias** relative to uniform exploration.
- **Time frame:** Weekly submissions over the capstone period; repository currently includes results through **`public/data/results/week_1` … `week_13`**. Oracle behaviour is **deterministic**: the same **x** yields the same **y** (no explicit observation noise in the logged outputs).

## Preprocessing / cleaning / labelling

- **Transformations on stored data:** **None** applied to the raw query–response logs before saving. Values are kept as returned by the oracle.
- **Model-side processing:** Surrogates may apply internal scaling (e.g. `StandardScaler` in SVR) or normalisation inside GP fitting; this does **not** overwrite the saved dataset.
- **Raw vs processed:** **Raw** cumulative arrays are preserved in `.npy` checkpoints; weekly `.txt` files are append-only portal-style logs.

## Uses

- **Appropriate uses:** Teaching and benchmarking **sequential black-box optimisation** under tiny budgets; reproducing or critiquing a documented BO workflow; studying failure modes (e.g. over-exploitation, narrow peaks, boundary optima).
- **Inappropriate uses:** **Do not** use this dataset to train or validate models for **high-stakes real-world decisions** (credit, hiring, clinical, safety-critical control). The objectives are **synthetic** and the sample distribution is **not** representative of any real population or physical process.
- **Composition and collection caveats for consumers:** Heavy **exploitation bias** and **function-specific** query policies mean the point cloud is **not** a uniform or unbiased map of **[0,1]^d**. Cross-function comparison of raw y values is not meaningful without normalisation. Consumers should read the **strategy reports** and **model card** alongside the numbers.

## Distribution

- **Availability:** Stored in this **GitHub repository** (course submission; visibility depends on repository settings — often private to the student and instructors).
- **Terms of use / licence:** **Academic / coursework use** unless the repository owner adds an explicit open licence. The hidden oracle definitions remain course intellectual property; this datasheet describes **observed inputs and outputs** and **methodology**, not the underlying function formulas.

## Maintenance

- **Maintainer:** The student author for the duration of the capstone and any agreed retention period afterward.
- **Updates:** New weeks are added by running `notebooks/data_management.ipynb` after each portal response, then committing updated `results/week_*` and optional `function_*/week_*_*.npy` files.
- **Versioning:** Week index in folder names (`week_1`, …) and optional strategy reports under `notes/` provide a coarse audit trail of what policy produced which batch.
