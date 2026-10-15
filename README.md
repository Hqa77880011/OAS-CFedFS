# OAS-CFedFS

Implementation of **Open-client federated feature selection with schema-aware confidence calibration under distribution shift.**

## Install

Use Python 3.11 in an isolated environment. From the repository root:

```bash
python -m pip install -e .
```

## Data

Input CSV columns are `client,split,y`, descriptor columns named in `descriptors`, and numeric feature columns. Source rows use `source`; every unseen client supplies disjoint unlabeled `cal` and labeled `test` rows. Entirely NaN client-feature columns denote unavailable schema; intermittent NaNs denote sample missingness. Common names must denote common measurements/units. The paper uses Gas Drift, HHAR, UCI HAR, PAMAP2, Madelon and Gisette from [UCI](https://archive.ics.uci.edu/). Keep natural client identities; identify constructed pseudo-clients explicitly. Known propensity arrays follow sorted source-client and feature-column order.

## Workflow

```bash
python -m oas_cfedfs prepare --input data/clients.csv
python -m oas_cfedfs fit
python -m oas_cfedfs run
python -m oas_cfedfs evaluate
python -m oas_cfedfs plot
```

Main commands accept `--config config.json` and `--output runs/default`. Use a separate output directory for each seed and condition. Its `settings.json` records the resolved configuration.

## Configuration

`minimum_propensity`, `folds` and `propensity_sensitivity` control availability correction. `alpha`, `bootstrap` and `minimum_observed` control evidence bounds. Coavailability/correlation/direction thresholds admit proxy edges. `pool_size`, `subset_size`, `feature_costs` and `budget` constrain selection. OOD and utility weights govern personalization, quality, cost and redundancy. Common training fields `epochs`, `batch_size`, `learning_rate`, `width` and `seed` control optimization, model size and randomness where present. Inspect `config.json` for defaults.

## Outputs

`summary.npz` contains source relevance, lower bounds, profiles, proxy graph and OOD threshold. `selections.json` records subsets, costs, gate values, communication and timing. `client_metrics.csv` contains unseen-client outcomes; `metrics.csv` adds lower-decile F1 and negative transfer against global truncation.

## Method and comparisons

Cross-fitted logistic availability models weight entropy-normalized mutual information. Proxy confidence uses a client-bootstrap maximum deviation after Fisher transformation. Profiles include masks, missingness, median/IQR and a rank-correlation sketch. OOD calibration leaves out whole source clients. Only source labels fit the selected-feature classifier. An empty conservative subset uses the source class prior.

Controls: `full`, `complete-case`, `no-ipw`, `no-lcb`, `no-proxy`, `no-ood`, `global-truncation`, `global-refill`. The run command evaluates these controls together. Changes to trainable objectives or architectures require a separate training run.

The `oas_cfedfs.compare` command averages repeated rows within independent units, then reports means, bootstrap intervals, paired Wilcoxon comparisons, rank-biserial effects and Holm adjustment per metric. For example:

```bash
python -m oas_cfedfs.compare --inputs "runs/*/client_metrics.csv" --reference full --metrics macro_f1 balanced_accuracy --unit client
```

