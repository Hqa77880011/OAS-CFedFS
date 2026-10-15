import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


def profile(x):
    available = np.isfinite(x).any(0)
    median = np.array([np.nanmedian(x[:, j]) if available[j] else 0 for j in range(x.shape[1])])
    spread = np.array([np.nanquantile(x[:, j], .75) - np.nanquantile(x[:, j], .25) if available[j] else 0 for j in range(x.shape[1])])
    missing = 1 - np.isfinite(x).mean(0)
    filled = np.where(np.isfinite(x), x, median[None])
    rank = pd.DataFrame(filled).rank().to_numpy()
    rank -= rank.mean(0)
    rank /= np.linalg.norm(rank, axis=0, keepdims=True).clip(1e-8)
    count = min(3, min(rank.shape))
    _, singular, right = randomized_svd(rank, n_components=count, random_state=13)
    sketch = np.pad(singular[:, None] * right, ((0, 3 - count), (0, 0)))
    return np.vstack([available, median, spread, missing, sketch])


def distance(a, b, scale):
    common = (a[0] > 0) & (b[0] > 0)
    union = (a[0] > 0) | (b[0] > 0)
    jaccard = 1 - common.sum() / max(1, union.sum())
    if not common.any():
        return 1e6
    difference = ((a[1:] - b[1:]) / scale[1:])[:, common]
    return float(jaccard + np.linalg.norm(difference) / np.sqrt(difference.size))


def prepare(c, out, input_path):
    frame = pd.read_csv(input_path)
    required = {'client', 'split', 'y'}
    if not required <= set(frame):
        raise ValueError('Input needs client, split, y plus numeric feature columns')
    features = [v for v in frame.columns if v not in required and v not in c['descriptors']]
    if not features:
        raise ValueError('No feature columns')
    sources = set(frame[frame.split == 'source'].client)
    targets = set(frame[frame.split.isin(['cal', 'test'])].client)
    if sources & targets:
        raise ValueError('Source and unseen client identities must be disjoint')
    frame.to_csv(out / 'data.csv', index=False)
    (out / 'features.json').write_text(json.dumps(features))


def fit(c, out):
    data = pd.read_csv(out / 'data.csv')
    names = json.loads((out / 'features.json').read_text(encoding='utf-8'))
    source = data[data.split == 'source']
    clients = list(source.groupby('client', sort=True))
    e, p = len(clients), len(names)
    if e < 3:
        raise ValueError('At least three source clients required for cross-fitting')
    relevance = np.zeros((e, p))
    profiles, descriptor = [], []
    for i, (_, g) in enumerate(clients):
        x = g[names].to_numpy(float)
        prof = profile(x)
        available = prof[0].astype(bool)
        profiles.append(prof)
        label_entropy = entropy(np.bincount(g.y.astype(int)))
        for j in np.flatnonzero(available):
            good = np.isfinite(x[:, j])
            if good.sum() >= 5 and len(np.unique(g.y[good])) > 1:
                relevance[i, j] = np.clip(mutual_info_classif(x[good, j:j + 1], g.y[good], random_state=c['seed'])[0] / max(label_entropy, 1e-8), 0, 1)
        descriptor.append(g[c['descriptors']].iloc[0].to_numpy(float) if c['descriptors'] else np.array([1.]))
    profiles = np.array(profiles)
    observed = profiles[:, 0]
    propensity = np.zeros_like(observed)
    h = np.array(descriptor)
    for fit_ids, held_ids in KFold(min(c['folds'], e), shuffle=True, random_state=c['seed']).split(h):
        for j in range(p):
            labels = observed[fit_ids, j]
            if len(np.unique(labels)) == 1:
                propensity[held_ids, j] = labels.mean()
            else:
                learner = LogisticRegression().fit(h[fit_ids], labels)
                propensity[held_ids, j] = learner.predict_proba(h[held_ids])[:, 1]
    if c['known_propensity']:
        propensity = np.load(c['known_propensity'])
        if propensity.shape != observed.shape:
            raise ValueError('Known propensity shape must be source clients by features')
    z = relevance * observed / np.maximum(propensity, c['minimum_propensity'])
    radius = np.sqrt(2 * z.var(0, ddof=1) * np.log(2 * p / c['alpha']) / e)
    radius += 7 / c['minimum_propensity'] * np.log(2 * p / c['alpha']) / (3 * (e - 1))
    lower = np.maximum(0, z.mean(0) - radius - c['propensity_sensitivity'])
    counts = observed.sum(0).clip(1)
    complete_case = relevance.sum(0) / counts
    variance = ((relevance - complete_case[None]) ** 2 * observed).sum(0) / (counts - 1).clip(1)
    cc_radius = np.sqrt(2 * variance * np.log(2 * p / c['alpha']) / counts) + 7 * np.log(2 * p / c['alpha']) / (3 * (counts - 1).clip(1))
    cc_lower = np.maximum(0, complete_case - cc_radius)
    eligible = np.flatnonzero(observed.sum(0) >= c['minimum_observed'])
    pools = [eligible[np.argsort(-score[eligible], kind='stable')[:c['pool_size']]] for score in [lower, complete_case, cc_lower, z.mean(0)]]
    graph_pool = np.unique(np.concatenate(pools))
    if not len(graph_pool):
        raise ValueError('No feature meets minimum source-client coverage')
    correlations = np.array([g[names].iloc[:, graph_pool].corr(method='spearman', min_periods=5).fillna(0).to_numpy() for _, g in clients])
    graph_observed = observed[:, graph_pool]
    co = graph_observed[:, :, None] * graph_observed[:, None, :]
    fisher = np.arctanh(np.clip(correlations, -.999, .999))
    mean = (fisher * co).sum(0) / co.sum(0).clip(1)
    rng = np.random.default_rng(c['seed'])
    deviations = []
    for _ in range(c['bootstrap']):
        ids = rng.integers(e, size=e)
        boot = (fisher[ids] * co[ids]).sum(0) / co[ids].sum(0).clip(1)
        deviations.append(abs(boot - mean).max())
    confidence = np.tanh(np.maximum(0, abs(mean) - np.quantile(deviations, 1 - c['alpha'])))
    agreement = np.maximum(((correlations > 0) * co).sum(0), ((correlations < 0) * co).sum(0)) / co.sum(0).clip(1)
    valid = (co.mean(0) >= c['minimum_coavailability']) & (confidence >= c['minimum_correlation']) & (agreement >= c['direction_agreement'])
    proxy = confidence * agreement * np.sqrt(co.mean(0)) * valid
    np.fill_diagonal(proxy, 0)
    scale = profiles.std(0).clip(1e-6)
    loo = []
    for i in range(e):
        other = np.arange(e) != i
        local_scale = profiles[other].std(0).clip(1e-6)
        loo.append(min(distance(profiles[i], p, local_scale) for p in profiles[other]))
    np.savez_compressed(out / 'summary.npz', relevance=relevance, observed=observed, lower=lower,
                        ipw=z.mean(0), cc_lower=cc_lower, graph_pool=graph_pool, profiles=profiles, scale=scale, proxy=proxy,
                        correlation=abs(np.tanh(mean)), ood_threshold=np.quantile(loo, 1 - c['ood_alpha']))


def select(summary, target, c, method):
    lower = summary['lower'].copy()
    if method == 'complete-case':
        lower = summary['relevance'].sum(0) / summary['observed'].sum(0).clip(1)
    elif method == 'no-ipw':
        lower = summary['cc_lower'].copy()
    elif method == 'no-lcb':
        lower = summary['ipw'].copy()
    eligible = summary['observed'].sum(0) >= c['minimum_observed']
    pool = np.flatnonzero(eligible)[np.argsort(-lower[eligible], kind='stable')[:c['pool_size']]]
    available = target[0].astype(bool)
    feasible = [j for j in pool if available[j]]
    distances = np.array([distance(target, p, summary['scale']) for p in summary['profiles']])
    gate = max(0., 1 - distances.min() / max(float(summary['ood_threshold']), 1e-8))
    if method == 'no-ood':
        gate = 1.
    weights = softmax(-distances / c['distance_temperature'])
    personalized = (weights[:, None] * summary['observed'] * summary['relevance']).sum(0) / (weights[:, None] * summary['observed']).sum(0).clip(1e-8)
    missing = [j for j in pool if not available[j]]
    graph_pool = summary['graph_pool']
    graph_index = {int(feature): i for i, feature in enumerate(graph_pool)}
    reward = np.zeros_like(lower)
    if missing:
        rows = [graph_index[int(j)] for j in missing]
        reward[graph_pool] = (lower[missing, None] * summary['proxy'][rows]).max(0)
    if method == 'no-proxy':
        reward[:] = 0
    cost = np.array(c['feature_costs'] or [1.] * len(lower))
    score = lower - c['cost_weight'] * cost + c['proxy_weight'] * reward + gate * (personalized + c['quality_weight'] * (1 - target[3]))
    selected = []
    if method in ['global-truncation', 'global-refill']:
        ranked = sorted(pool, key=lambda j: -lower[j])
        if method == 'global-truncation':
            ranked = ranked[:c['subset_size']]
        feasible = [j for j in ranked if available[j]]
        score = lower.copy()
    while len(selected) < c['subset_size']:
        allowed = [j for j in feasible if j not in selected and cost[selected].sum() + cost[j] <= c['budget']]
        if not allowed:
            break
        gain = [score[j] - c['redundancy_weight'] * summary['correlation'][graph_index[int(j)], [graph_index[int(v)] for v in selected]].sum() for j in allowed]
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        selected.append(allowed[best])
    return selected, gate, cost[selected].sum()


def run(c, out):
    data = pd.read_csv(out / 'data.csv')
    names = json.loads((out / 'features.json').read_text(encoding='utf-8'))
    summary = dict(np.load(out / 'summary.npz'))
    source = data[data.split == 'source']
    predictions, selections = [], []
    for client, target in data[data.split != 'source'].groupby('client'):
        cal = target[target.split == 'cal']
        test = target[target.split == 'test']
        if not len(cal) or not len(test):
            raise ValueError('Each unseen client needs disjoint cal and test records')
        for method in ['full', 'complete-case', 'no-ipw', 'no-lcb', 'no-proxy', 'no-ood', 'global-truncation', 'global-refill']:
            start = time.perf_counter()
            target_profile = profile(cal[names].to_numpy(float))
            indices, gate, cost = select(summary, target_profile, c, method)
            elapsed = 1000 * (time.perf_counter() - start)
            if indices:
                columns = [names[j] for j in indices]
                learner = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000))
                learner.fit(source[columns], source.y.astype(int))
                predicted = learner.predict(test[columns])
            else:
                predicted = np.full(len(test), source.y.mode().iloc[0])
            for row_id, label in zip(test.index, predicted):
                predictions.append(dict(method=method, client=client, row=int(row_id), prediction=int(label)))
            selections.append(dict(method=method, client=client, indices=indices, gate=gate, cost=float(cost),
                                   utilization=len(indices) / c['subset_size'], selection_ms=elapsed,
                                   communication_bytes=int(target_profile.nbytes)))
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    (out / 'selections.json').write_text(json.dumps(selections))


def evaluate(c, out):
    raw = pd.read_csv(out / 'data.csv')
    predicted = pd.read_csv(out / 'predictions.csv')
    selections = pd.DataFrame(json.loads((out / 'selections.json').read_text(encoding='utf-8')))
    rows = []
    for (method, client), group in predicted.groupby(['method', 'client']):
        y = raw.loc[group.row, 'y'].astype(int)
        result = dict(method=method, client=client, macro_f1=f1_score(y, group.prediction, average='macro', zero_division=0),
                      balanced_accuracy=balanced_accuracy_score(y, group.prediction))
        detail = selections[(selections.method == method) & (selections.client == client)].iloc[0]
        for key in ['gate', 'cost', 'utilization', 'selection_ms', 'communication_bytes']:
            result[key] = detail[key]
        rows.append(result)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'client_metrics.csv', index=False)
    baseline = metrics[metrics.method == 'global-truncation'].set_index('client').macro_f1
    summary = []
    for method, g in metrics.groupby('method'):
        result = g.mean(numeric_only=True).to_dict()
        result.update(method=method, tenth_percentile_f1=g.macro_f1.quantile(.1),
                      negative_transfer=float((g.set_index('client').macro_f1 < baseline).mean()))
        summary.append(result)
    pd.DataFrame(summary).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'client_metrics.csv')
    fig, ax = plt.subplots(figsize=(9, 4))
    d.boxplot(column='macro_f1', by='method', ax=ax, rot=35)
    fig.suptitle('')
    ax.set(title='', ylabel='Unseen-client Macro-F1')
    fig.tight_layout()
    fig.savefig(out / 'clients.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'fit', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
