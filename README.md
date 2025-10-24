# MoDE‑Boost Framework

[![License: BSD‑3‑Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

The **MoDE‑Boost** framework provides a end‑to‑end pipeline for extracting features from shared‑mobility bike‑trip data and training both regression and classification models (the **MoDE‑Boost** framework).  It is written for Python 3.11 and is fully reproducible.

---

## Table of Contents

- [MoDE‑Boost Framework](#modeboost-framework)
  - [Table of Contents](#table-of-contents)
  - [1. Project Structure](#1-project-structure)
  - [2. Overview](#2-overview)
  - [3. Usage](#3-usage)
    - [3.1. Data fetching](#31-data-fetching)
    - [3.2. Data processing](#32-data-processing)
      - [3.2.1. NL datasets](#321-nl-datasets)
      - [3.2.2. US datasets (CitiNYC / Divvy)](#322-us-datasets-citinyc--divvy)
    - [3.3. Training \& evaluation](#33-training--evaluation)
  - [1.5. License](#15-license)

---

## 1. Project Structure

```text
project-root/
├─ data/
│   ├─ external/          # External data sources (unchanged by the project)
│   ├─ raw/               # Original, untouched data
│   ├─ interim/           # Intermediate, cleaned data
│   ├─ processed/         # Final ML‑ready datasets
│   └─ make_dataset.py    # Optional preprocessing pipelines
├─ docs/                  # Documentation files
├─ logs/                  # Log files generated at runtime
├─ models/                # Trained model artifacts
├─ reports/               # Generated reports, figures and benchmarks
│   ├─ __init__.py
│   ├─ benchmarks/        # Benchmark results
│   └─ figures/           # Plots and visualisations
├─ src/
│   ├─ __init__.py
│   ├─ config/            # Global configuration helpers
│   │   ├─ __init__.py
│   │   ├─ env.py         # Loads environment variables from `.env`
│   │   ├─ constants.py   # Core constants used throughout the repo
│   │   └─ paths.py       # Centralised pathlib definitions
│   ├─ components/        # Feature‑engineering modules
│   │   ├─ __init__.py
│   │   ├─ fourier_transformer.py
│   │   ├─ group_transformer.py
│   │   ├─ model_factory.py
│   │   ├─ rbf_transformer.py
│   │   ├─ temporal_extractor.py
│   │   └─ traffic_adjuster.py
│   ├─ data/              # Data loading & preprocessing utilities
│   │   ├─ __init__.py
│   │   ├─ cleaner/
│   │   │   ├─ __init__.py
│   │   │   ├─ core.py          # Abstract cleaning interfaces
│   │   │   ├─ models.py        # Enums / config classes for cleaners
│   │   │   └─ bike_cleaner.py  # Concrete implementation for bike data
│   │   ├─ loader/
│   │   │   ├─ __init__.py
│   │   │   ├─ core.py          # Abstract loader interfaces
│   │   │   ├─ models.py        # Enums / config classes for loaders
│   │   │   └─ bike_loader.py   # Loads raw bike‑trip data from S3
│   │   ├─ bike_processor.py    # Transforms raw trips into hourly demand series
│   │   ├─ preprocess.py        # Additional preprocessing helpers
│   │   ├─ request.py           # API request utilities
│   │   └─ prepare_data.py      # End‑to‑end pipeline that produces parquet files
│   ├─ model/                   # Model definition & training scripts
│   │   ├─ __init__.py
│   │   ├─ ggdpb/               # MoDE‑Boost implementation
│   │   │   ├─ __init__.py
│   │   │   ├─ forecaster.py    # Forecasting pipeline implementation
│   │   │   └─ models.py        # Configuration objects & enums used by the pipeline
│   │   ├─ train.py             # Regression training
│   │   ├─ train_classif.py     # Classification training
│   │   ├─ train_optim.py       # Hyper‑parameter optimisation (Optuna)
│   │   └─ predict.py           # Model inference & evaluation utilities
│   └─ utils/                   # General helper modules
│       ├─ __init__.py
│       ├─ helper.py
│       ├─ models.py
│       └─ logger.py        # Loguru‑based logger configuration
├─ .env                     # Environment variables (see``.env.example`)
└─ pyproject.toml           # Packaging and dependency definition
```

---

## 2. Overview

The repository ships two primary components:

1. **Feature‑extraction pipeline** – transforms raw bike‑trip records into a rich set of temporal, spatial and network‑based features.
2. **MoDE‑Boost framework** – a unified modelling interface that supports both **regression** (exact demand) and **classification** (demand levels: *Low*, *Medium*, *High*).

The framework has been evaluated on five metropolitan areas (New York, Chicago, Amsterdam, The Hague, Rotterdam) and the results are summarised in the `reports/` folder.

---

## 3. Usage

### 3.1. Data fetching

Raw bike‑trip data can be downloaded using the loader script:

```bash
python src/data/loader/bike_loader.py \
    --dataset <citi|divvy> \
    --bucket_name <tripdata|divvy-tripdata> \
    --year <2021|2022>
```

* `--dataset` – source provider (Citi Bike or Divvy).
* `--bucket_name` – corresponding S3 bucket.
* `--year` – year of interest.

For the Dutch datasets, download the archives from the provided cloud location and place them under `data/interim/<city>/` (e.g., `data/interim/amsterdam/`).

---

### 3.2. Data processing

The processing steps differ slightly between the US and NL data sources.

#### 3.2.1. NL datasets

```bash
python src/data/prepare_data.py \
    --city amsterdam \
    --output data/processed/amsterdam.parquet
```

All required arguments are documented in the script's ``--help`` output.

#### 3.2.2. US datasets (CitiNYC / Divvy)

1. **Cleaning** – removes invalid rows and normalises timestamps.

   ```bash
   python src/data/cleaner/bike_cleaner.py --dataset citi
   ```

2. **Transformation** – aggregates trips into an hourly demand time‑series.

   ```bash
   python src/data/bike_processor.py --dataset citi
   ```

3. **Final preparation** – creates the ML‑ready parquet file.

   ```bash
   python src/data/prepare_data.py --city new_york --output data/processed/new_york.parquet
   ```

> **Tip**: All scripts write their outputs to the paths defined in `src/config/paths.py`, ensuring reproducibility.

---

### 3.3. Training & evaluation

Once the processed data are available, training is straightforward:

```bash
python src/model/train.py ...
```

The script will:

1. Train the selected model.
2. Persist the model artifact under `models/`.
3. Generate predictions on a hold‑out set and store evaluation metrics in `reports/benchmarks/`.

Below we provide the CLI arguments for `src/model/train.py`

| Flag | Parameter | Type | Default | Allowed values | Description |
|------|-----------|------|---------|----------------|-------------|
| `-d`, `--dataset` | `dataset` | `str` | — | — | Name of the dataset to load (e.g., `nyc`, `amsterdam`). |
| `-w`, `--window` | `window` | `int` | — | — | Size of the sliding window (in hours) used when aggregating raw bike-trip data. |
| `-f`, `--fh` | `fh` | `int` | — | — | Forecast horizon (number of steps ahead) for the model. |
| `-s`, `--suffix` | `suffix` | `str` | `v1` | — | Suffix added to generated files; useful for versioning outputs. |
| `-e`, `--extension` | `extension` | `str` | — | — | File extension to look for when reading raw data (e.g., `csv`, `parquet`). |
| `-t`, `--target` | `target` | `str` (multiple) | — | — | One or more target columns to predict; can be supplied as a space-separated list (`nargs='+'`). |
| `--return_X_y` | `return_X_y` | `store_true` | `False` | — | Return separate feature matrix **X** and target **y** during data loading. |
| `--dropna` | `dropna` | `store_true` | `False` | — | Drop rows with `NaN` values during the data-loading step. |
| `-g`, `--group_col` | `group_col` | `str` | — | — | Column used to group the data (e.g., city district or station identifier). |
| `-D`, `--device` | `device` | `str` | — | `cpu`, `cuda` | Device on which to train the model (CPU or CUDA GPU). |
| `-T`, `--task` | `task` | `str` | — | `regression`, `classification` | Type of learning problem; selects the appropriate estimator. |

---

For hyper‑parameter optimisation, run:

```bash
python src/model/optuna.py 
```

---

## 1.5. License

The project is licensed under the **BSD 3‑Clause License**. See the `LICENSE` file for the full text.

---

*If you have questions or need assistance, please open an issue or contact the maintainers.*
