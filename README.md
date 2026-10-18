# OAS-CFedFS

Implementation of **Open-client federated feature selection with schema-aware confidence calibration under distribution shift.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Input CSV columns are `client,split,y`, descriptor columns named in `descriptors`, and numeric feature columns. Source rows use `source`; every unseen client supplies disjoint unlabeled `cal` and labeled `test` rows. Entirely NaN client-feature columns denote unavailable schema; intermittent NaNs denote sample missingness. Common names must denote common measurements/units. The paper uses Gas Drift, HHAR, UCI HAR, PAMAP2, Madelon and Gisette from [UCI](https://archive.ics.uci.edu/). Keep natural client identities; identify constructed pseudo-clients explicitly. Known propensity arrays follow sorted source-client and feature-column order.

```bash
python -m oas_cfedfs prepare --input data/clients.csv --config config.json --output runs/full
```

## Training and inference

```bash
python -m oas_cfedfs fit --config config.json --output runs/full
python -m oas_cfedfs run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m oas_cfedfs evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `complete-case`, `no-ipw`, `no-lcb`, `no-proxy`, `no-ood`, `global-truncation`, `global-refill`.

The run command evaluates these controls together.

## Outputs

All main outputs are written below `runs/full/`.

`summary.npz` contains source relevance, lower bounds, profiles, proxy graph and OOD threshold. `selections.json` records subsets, costs, gate values, communication and timing. `client_metrics.csv` contains unseen-client outcomes; `metrics.csv` adds lower-decile F1 and negative transfer against global truncation.

```bash
python -m oas_cfedfs plot --config config.json --output runs/full
```
