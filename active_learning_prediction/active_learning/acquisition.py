# -*- coding: utf-8 -*-
"""Acquisition strategies including iterative frequency-rank and frequency-value."""
"""Acquisition strategies: random, uncertainty, similarity, frequency_value and frequency_rank."""
import math
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity

class Acquisition:
    def __init__(self, method: str, seed: int = 42, **kwargs):
        self.acquisition_method = {
            'random': self.random_pick,
            'exploration': greedy_exploration,
            'exploitation': greedy_exploitation,
            'dynamic': dynamic_exploration,
            'dynamicbald': dynamic_exploration_bald,
            'bald': bald,
            'similarity_jaccard': similarity_search_jaccard,
            'similarity_cosine': similarity_search_cosine,
            'frequency_value': frequency_value_search,
            'frequency_rank': frequency_rank_search
        }
        if method not in self.acquisition_method: raise ValueError(f'Unknown acquisition method {method!r}; choices: {list(self.acquisition_method)}')
        self.method = method
        self.params = dict(kwargs)
        if method in ('dynamic', 'dynamicbald'): self.params['lambd'] = 0.90
        self.rng = np.random.default_rng(seed)
        self.iteration = 0
        self.last_selected_idx = np.empty(0, dtype=int)
        self.last_exploit_idx = np.empty(0, dtype=int)
        self.last_explore_idx = np.empty(0, dtype=int)
        self.last_selected_sources = np.empty(0, dtype=object)
        self.last_acquisition_details = {}
    def acquire(self, logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1) -> np.ndarray:
        self.iteration += 1
        pool_idx = to_1d_index_array(pool_idx)
        self.last_selected_idx = np.empty(0, dtype=int)
        self.last_exploit_idx = np.empty(0, dtype=int)
        self.last_explore_idx = np.empty(0, dtype=int)
        self.last_selected_sources = np.empty(0, dtype=object)
        self.last_acquisition_details = {}
        if len(pool_idx) != len(np.unique(pool_idx)): raise ValueError('pool_idx contains duplicate sample indices')
        if n < 0: raise ValueError('n must be nonnegative')
        n = min(int(n), len(pool_idx))
        if n == 0: return np.empty(0, dtype=int)
        if logits_N_K_C is not None and logits_N_K_C.shape[0] != len(pool_idx): raise ValueError('logits_N_K_C first dimension must equal len(pool_idx)')
        params = dict(self.params)
        dynamic_record = {}
        params.update(iteration=self.iteration, rng=self.rng, dynamic_record=dynamic_record, df_screen=getattr(self, 'df_screen', None), feature_cols=getattr(self, 'feature_cols', None), train_idx=getattr(getattr(self, 'handler', None), 'train_idx', None))
        selected = self.acquisition_method[self.method](logits_N_K_C=logits_N_K_C, pool_idx=pool_idx, n=n, **params)
        selected = to_1d_index_array(selected)
        if len(selected) != n: raise AssertionError(f'{self.method} selected {len(selected)}, expected {n}')
        if len(np.unique(selected)) != n: raise AssertionError(f'{self.method} returned duplicate picks')
        if not np.isin(selected, pool_idx).all(): raise AssertionError(f'{self.method} picked indices outside pool_idx')
        self.last_selected_idx = selected.copy()
        if self.method in ('dynamic', 'dynamicbald'):
            self.last_exploit_idx = dynamic_record['exploit_idx'].copy()
            self.last_explore_idx = dynamic_record['explore_idx'].copy()
            second_stage = 'bald' if self.method == 'dynamicbald' else 'exploration'
            self.last_selected_sources = np.array(['exploitation'] * len(self.last_exploit_idx) + [second_stage] * len(self.last_explore_idx), dtype=object)
            self.last_acquisition_details = {'cycle': self.iteration, 'lambda': dynamic_record['lambd'], 'n_exploit': len(self.last_exploit_idx), 'n_explore': len(self.last_explore_idx), 'second_stage': second_stage}
            if not np.array_equal(selected, np.concatenate([self.last_exploit_idx, self.last_explore_idx])): raise AssertionError('Dynamic source indices and returned indices disagree')
        else:
            self.last_selected_sources = np.full(len(selected), self.method, dtype=object)
        print(f'[acquire] method={self.method}, iteration={self.iteration}, selected={len(selected)}/{n}, unique=PASS')
        return selected
    def __call__(self, *args, **kwargs): return self.acquire(*args, **kwargs)
    def random_pick(self, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
        return self.rng.choice(pool_idx, size=n, replace=False).astype(int)

def to_1d_index_array(x) -> np.ndarray:
    if x is None: return np.empty(0, dtype=int)
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=int).reshape(-1)

def safe_concat_picks(picks) -> np.ndarray:
    arrays = [to_1d_index_array(x) for x in picks]
    arrays = [x for x in arrays if len(x)]
    return np.concatenate(arrays).astype(int) if arrays else np.empty(0, dtype=int)

def rank_desc_random_ties(scores, n: int, rng=None) -> np.ndarray:
    if rng is None: rng = np.random.default_rng(42)
    if isinstance(scores, torch.Tensor): scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).reshape(-1)
    return np.lexsort((rng.random(len(scores)), -scores))[:n]

def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])

def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)

def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    return torch.mean(entropy(logits_N_K_C, dim=dim, keepdim=keepdim), dim=1)

def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    return (entropy(logit_mean(logits_N_K_C, dim=1), dim=-1) - mean_sample_entropy(logits_N_K_C)).clamp_min(0)

def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True):
    mean_probs = torch.mean(torch.exp(logits_N_K_C), dim=1)
    pred = mean_probs if return_prob else torch.argmax(mean_probs, dim=1)
    return (pred, mean_sample_entropy(logits_N_K_C)) if return_uncertainty else pred

def greedy_exploitation(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    if logits_N_K_C is None: raise ValueError('exploitation requires logits_N_K_C')
    scores = torch.exp(logits_N_K_C).mean(dim=1)[:, 1]
    return pool_idx[rank_desc_random_ties(scores, n, kwargs.get('rng'))]

def greedy_exploration(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    if logits_N_K_C is None: raise ValueError('exploration requires logits_N_K_C')
    scores = mean_sample_entropy(logits_N_K_C)
    return pool_idx[rank_desc_random_ties(scores, n, kwargs.get('rng'))]

def bald(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    if logits_N_K_C is None: raise ValueError('BALD requires logits_N_K_C')
    scores = mutual_information(logits_N_K_C)
    return pool_idx[rank_desc_random_ties(scores, n, kwargs.get('rng'))]

def _dynamic(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int, lambd: float, iteration: int, exploration_fn, rng=None, dynamic_record=None) -> np.ndarray:
    if not 0 < lambd < 1: raise ValueError('lambd must be in (0, 1)')
    if iteration < 1: raise ValueError('iteration must start from 1')
    if logits_N_K_C is None: raise ValueError('dynamic strategies require logits_N_K_C')
    factor = min(1.0, max(0.0, lambd ** (-iteration) - 1.0))
    n_exploit = max(0, min(n, int(round(n * factor))))
    n_explore = n - n_exploit
    exploit = to_1d_index_array(greedy_exploitation(logits_N_K_C, pool_idx, n_exploit, rng=rng)) if n_exploit else np.empty(0, dtype=int)
    mask = ~np.isin(pool_idx, exploit)
    remaining_pool = pool_idx[mask]
    remaining_logits = logits_N_K_C[torch.as_tensor(mask, dtype=torch.bool, device=logits_N_K_C.device)]
    explore = to_1d_index_array(exploration_fn(remaining_logits, remaining_pool, n_explore, rng=rng)) if n_explore else np.empty(0, dtype=int)
    selected = safe_concat_picks([exploit, explore])
    unique_count = len(np.unique(selected))
    if len(selected) != n or unique_count != n: raise AssertionError('dynamic acquisition returned wrong count or duplicated indices')
    if dynamic_record is not None:
        dynamic_record.update(exploit_idx=exploit.copy(), explore_idx=explore.copy(), lambd=lambd)
    strategy = 'BALD' if exploration_fn is bald else 'Exploration'
    method = 'DYNAMIC-BALD' if exploration_fn is bald else 'DYNAMIC'
    print(f'[{method}] Cycle={iteration} | lambda={lambd:.2f} | Exploitation={len(exploit)} | {strategy}={len(explore)} | Total={len(selected)} | Unique={unique_count} | PASS', flush=True)
    return selected

def dynamic_exploration(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, lambd: float = 0.90, iteration: int = 1, **kwargs) -> np.ndarray:
    return _dynamic(logits_N_K_C, pool_idx, n, lambd, iteration, greedy_exploration, rng=kwargs.get('rng'), dynamic_record=kwargs.get('dynamic_record'))

def dynamic_exploration_bald(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, lambd: float = 0.90, iteration: int = 1, **kwargs) -> np.ndarray:
    return _dynamic(logits_N_K_C, pool_idx, n, lambd, iteration, bald, rng=kwargs.get('rng'), dynamic_record=kwargs.get('dynamic_record'))

def _get_hit_idx(df_screen, train_idx):
    if df_screen is None or train_idx is None: raise ValueError('This strategy requires Acquisition.df_screen and Acquisition.handler.train_idx')
    if 'y' not in df_screen.columns: raise KeyError('df_screen must contain binary training target column "y" (positive=1)')
    train_idx = to_1d_index_array(train_idx)
    return train_idx[df_screen.iloc[train_idx]['y'].to_numpy() == 1]

def _get_gene_cols(df_screen):
    normalized = {str(c).lower().replace(' ', '').replace('_', ''): c for c in df_screen.columns}
    a = normalized.get('genea', normalized.get('gene1'))
    b = normalized.get('geneb', normalized.get('gene2'))
    if a is None or b is None: raise KeyError('frequency requires Gene A and Gene B columns in df_screen')
    return a, b

def _frequency_gene_name(value):
    """Validate and normalize a gene identifier for frequency strategies."""
    if pd.isna(value):
        raise ValueError('Gene identifiers cannot be missing')
    gene = str(value).strip()
    if not gene:
        raise ValueError('Gene identifiers cannot be empty')
    return gene

def _prepare_frequency_context(pool_idx, n, kwargs):
    df_screen, train_idx = kwargs.get('df_screen'), kwargs.get('train_idx')
    rng = kwargs.get('rng')
    if rng is None:
        rng = np.random.default_rng(42)
    if not isinstance(df_screen, pd.DataFrame):
        raise TypeError('frequency requires df_screen to be a pandas DataFrame')
    if 'y' not in df_screen.columns:
        raise KeyError('frequency requires binary target column y (positive=1)')
    if train_idx is None:
        raise ValueError('frequency requires the current labeled train_idx')
    pool_idx = to_1d_index_array(pool_idx)
    train_idx = to_1d_index_array(train_idx)
    if len(np.unique(pool_idx)) != len(pool_idx):
        raise ValueError('frequency received duplicate candidate indices')
    if len(np.unique(train_idx)) != len(train_idx):
        raise ValueError('frequency received duplicate training indices')
    if np.intersect1d(pool_idx, train_idx).size:
        raise ValueError('frequency training and candidate indices overlap')
    if ((pool_idx.size and (pool_idx.min() < 0 or pool_idx.max() >= len(df_screen)))
            or (train_idx.size and (train_idx.min() < 0 or train_idx.max() >= len(df_screen)))):
        raise IndexError('frequency received indices outside df_screen')
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 0 or n > len(pool_idx):
        raise ValueError('frequency n must be an integer between 0 and candidate-pool size')
    n = int(n)
    if n == 0:
        return df_screen, pool_idx, n, None, None, {}, rng, [], []
    col_a, col_b = _get_gene_cols(df_screen)
    train = df_screen.iloc[train_idx]
    if not train['y'].isin([0, 1]).all():
        raise ValueError('frequency requires binary y values in the labeled training set')
    positive_pairs = train.loc[train['y'].eq(1), [col_a, col_b]]
    counts = Counter()
    for raw_a, raw_b in positive_pairs.itertuples(index=False, name=None):
        gene_a, gene_b = _frequency_gene_name(raw_a), _frequency_gene_name(raw_b)
        if gene_a == gene_b:
            raise ValueError('frequency does not support self-pairs')
        counts[gene_a] += 1
        counts[gene_b] += 1
    candidate = df_screen.iloc[pool_idx][[col_a, col_b]]
    genes_a = [_frequency_gene_name(value) for value in candidate[col_a].tolist()]
    genes_b = [_frequency_gene_name(value) for value in candidate[col_b].tolist()]
    if any(a == b for a, b in zip(genes_a, genes_b)):
        raise ValueError('frequency candidate pool contains self-pairs')
    return df_screen, pool_idx, n, col_a, col_b, counts, rng, genes_a, genes_b

def _select_frequency_by_scores(pool_idx, primary, n, rng, secondary=None):
    primary = np.asarray(primary, dtype=np.float64).reshape(-1)
    if len(primary) != len(pool_idx) or not np.isfinite(primary).all():
        raise ValueError('frequency primary scores have invalid length or non-finite values')
    if secondary is None:
        order = np.argsort(-primary, kind='stable')
    else:
        secondary = np.asarray(secondary, dtype=np.float64).reshape(-1)
        if len(secondary) != len(pool_idx) or not np.isfinite(secondary).all():
            raise ValueError('frequency secondary scores have invalid length or non-finite values')
        order = np.lexsort((-secondary, -primary))
    chosen = []
    start = 0
    while len(chosen) < n:
        stop = start + 1
        while stop < len(order):
            current, following = order[start], order[stop]
            same_score = primary[current] == primary[following]
            if secondary is not None:
                same_score = same_score and secondary[current] == secondary[following]
            if not same_score:
                break
            stop += 1
        tied_positions = order[start:stop]
        needed = min(n - len(chosen), len(tied_positions))
        chosen.extend(rng.choice(tied_positions, size=needed, replace=False).tolist())
        start = stop
    return pool_idx[np.asarray(chosen, dtype=int)]

def frequency_value_search(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    """Select by descending (max(fA, fB), min(fA, fB)) with random tie-breaking."""
    _, pool_idx, n, _, _, counts, rng, genes_a, genes_b = _prepare_frequency_context(pool_idx, n, kwargs)
    if n == 0:
        return np.empty(0, dtype=int)
    if not counts:
        return rng.choice(pool_idx, size=n, replace=False).astype(int)
    freq_a = np.fromiter((counts.get(gene, 0) for gene in genes_a), dtype=np.int64, count=len(pool_idx))
    freq_b = np.fromiter((counts.get(gene, 0) for gene in genes_b), dtype=np.int64, count=len(pool_idx))
    primary = np.maximum(freq_a, freq_b)
    secondary = np.minimum(freq_a, freq_b)
    return _select_frequency_by_scores(pool_idx, primary, n, rng, secondary=secondary)

def frequency_rank_search(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    """Select by descending rank_A + rank_B using average percentile ranks."""
    df_screen, pool_idx, n, col_a, col_b, counts, rng, genes_a, genes_b = _prepare_frequency_context(pool_idx, n, kwargs)
    if n == 0:
        return np.empty(0, dtype=int)
    if not counts:
        return rng.choice(pool_idx, size=n, replace=False).astype(int)
    universe = set()
    for raw_a, raw_b in df_screen[[col_a, col_b]].itertuples(index=False, name=None):
        gene_a, gene_b = _frequency_gene_name(raw_a), _frequency_gene_name(raw_b)
        if gene_a == gene_b:
            raise ValueError('frequency does not support self-pairs')
        universe.add(gene_a)
        universe.add(gene_b)
    genes = sorted(universe)
    if not genes:
        raise ValueError('frequency cannot rank an empty gene universe')
    frequencies = np.fromiter((counts.get(gene, 0) for gene in genes), dtype=np.int64, count=len(genes))
    sorted_frequencies = np.sort(frequencies)
    left = np.searchsorted(sorted_frequencies, frequencies, side='left')
    right = np.searchsorted(sorted_frequencies, frequencies, side='right')
    average_percentile_ranks = ((left + 1 + right) / 2.0) / len(genes)
    rank_map = dict(zip(genes, average_percentile_ranks))
    scores = np.fromiter((rank_map[a] + rank_map[b] for a, b in zip(genes_a, genes_b)),
                         dtype=np.float64, count=len(pool_idx))
    return _select_frequency_by_scores(pool_idx, scores, n, rng)

def tanimoto_binary_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return Jaccard similarities of shape (n_reference, n_candidates)."""
    x, y = np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)
    intersection = y @ x.T
    union = y.sum(axis=1, dtype=np.int64)[:, None] + x.sum(axis=1, dtype=np.int64)[None, :] - intersection
    return intersection / np.maximum(union, 1)

def similarity_search_jaccard(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    df_screen, feature_cols = kwargs.get('df_screen'), kwargs.get('feature_cols')
    rng = kwargs.get('rng')
    if rng is None: rng = np.random.default_rng(42)
    hit_idx = _get_hit_idx(df_screen, kwargs.get('train_idx'))
    if not len(hit_idx): return rng.choice(pool_idx, size=n, replace=False)
    if feature_cols is None: raise ValueError('similarity_jaccard requires feature_cols')
    hits = df_screen.iloc[hit_idx][feature_cols].to_numpy(dtype=np.int32)
    scores = np.empty(len(pool_idx), dtype=np.float64)
    block_size = int(kwargs.get('block_size', 512))
    if block_size < 1: raise ValueError('block_size must be positive')
    for start in range(0, len(pool_idx), block_size):
        end = min(start + block_size, len(pool_idx))
        pool = df_screen.iloc[pool_idx[start:end]][feature_cols].to_numpy(dtype=np.int32)
        scores[start:end] = tanimoto_binary_matrix(pool, hits).max(axis=0)
    return pool_idx[rank_desc_random_ties(scores, n, rng)]

def similarity_search_cosine(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    df_screen, feature_cols = kwargs.get('df_screen'), kwargs.get('feature_cols')
    rng = kwargs.get('rng')
    if rng is None: rng = np.random.default_rng(42)
    hit_idx = _get_hit_idx(df_screen, kwargs.get('train_idx'))
    if not len(hit_idx): return rng.choice(pool_idx, size=n, replace=False)
    if feature_cols is None: raise ValueError('similarity_cosine requires feature_cols')
    hits = df_screen.iloc[hit_idx][feature_cols].to_numpy(dtype=np.float32)
    scores = np.empty(len(pool_idx), dtype=np.float64)
    block_size = int(kwargs.get('block_size', 512))
    if block_size < 1: raise ValueError('block_size must be positive')
    for start in range(0, len(pool_idx), block_size):
        end = min(start + block_size, len(pool_idx))
        pool = df_screen.iloc[pool_idx[start:end]][feature_cols].to_numpy(dtype=np.float32)
        scores[start:end] = cosine_similarity(pool, hits).max(axis=1)
    return pool_idx[rank_desc_random_ties(scores, n, rng)]
