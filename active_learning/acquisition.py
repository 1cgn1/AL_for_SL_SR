import numpy as np
from torch import Tensor
import torch
import math
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
            'similarity_cosine': similarity_search_cosine
        }

        assert method in self.acquisition_method, \
            f"method must be one of {self.acquisition_method.keys()}"

        self.method = method
        self.params = kwargs
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

    def acquire(self, logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1) -> np.ndarray:
        self.iteration += 1

        selected_idx = self.acquisition_method[self.method](logits_N_K_C=logits_N_K_C,pool_idx=pool_idx,n=n,iteration=self.iteration,
                                                            df_screen=self.df_screen,feature_cols=self.feature_cols,train_idx=self.handler.train_idx,**self.params)

        selected_idx = to_1d_index_array(selected_idx)

        print(
            f"[DEBUG acquire] method={self.method}, "
            f"iteration={self.iteration}, "
            f"n={n}, "
            f"selected_type={type(selected_idx)}, "
            f"selected_shape={selected_idx.shape}, "
            f"selected_len={len(selected_idx)}"
        )

        return selected_idx

    def __call__(self, *args, **kwargs):
        return self.acquire(*args, **kwargs)

    def random_pick(self, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
        picks = self.rng.integers(0, len(pool_idx), n)
        return pool_idx[picks]

def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True):

    mean_probs_N_C = torch.mean(torch.exp(logits_N_K_C), dim=1)

    uncertainty = mean_sample_entropy(logits_N_K_C)

    if return_prob:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat

def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])

def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)

def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)
    return entropy_mean_N

def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)
    I = mean_entropy_N - entropy_mean_N
    return I

def to_1d_index_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=int)
    arr = np.asarray(x, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr

def safe_concat_picks(picks) -> np.ndarray:
    clean_picks = []
    for p in picks:
        p_arr = to_1d_index_array(p)
        if len(p_arr) > 0:
            clean_picks.append(p_arr)
    if len(clean_picks) == 0:
        return np.array([], dtype=int)
    if len(clean_picks) == 1:
        return clean_picks[0]
    return np.concatenate(clean_picks).astype(int)

def greedy_exploitation(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    mean_probs = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    picks = torch.argsort(mean_probs, descending=True)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

def greedy_exploration(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    picks = torch.argsort(entropy_mean_N, descending=True)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

def dynamic_exploration(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,lambd: float = 0.95,iteration: int = 0,**kwargs) -> np.ndarray:
    exploitation_factor = (1 / (lambd ** iteration)) - 1
    n_exploit = int(round(n * exploitation_factor))
    n_exploit = max(0, min(n_exploit, n, len(pool_idx)))
    n_explore = n - n_exploit
    picks = []
    if n_exploit > 0:
        picks.append(greedy_exploitation(logits_N_K_C, pool_idx, n_exploit))
    if n_explore > 0:
        picks.append(greedy_exploration(logits_N_K_C, pool_idx, n_explore))
    return safe_concat_picks(picks)

def dynamic_exploration_bald(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,lambd: float = 0.95,iteration: int = 0,**kwargs) -> np.ndarray:
    exploitation_factor = (1 / (lambd ** iteration)) - 1
    n_exploit = int(round(n * exploitation_factor))
    n_exploit = max(0, min(n_exploit, n, len(pool_idx)))
    n_explore = n - n_exploit
    picks = []
    if n_exploit > 0:
        picks.append(greedy_exploitation(logits_N_K_C, pool_idx, n_exploit))
    if n_explore > 0:
        picks.append(bald(logits_N_K_C, pool_idx, n_explore))
    return safe_concat_picks(picks)

def bald(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    I = mutual_information(logits_N_K_C)
    picks = torch.argsort(I, descending=False)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

def tanimoto_binary_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    intersection = np.dot(Y, X.T)
    Y_sum = Y.sum(axis=1)[:, None]
    X_sum = X.sum(axis=1)[None, :]
    return intersection / (Y_sum + X_sum - intersection + 1e-10)

def similarity_search_jaccard(pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    df_screen = kwargs["df_screen"]
    feature_cols = kwargs["feature_cols"]
    train_idx = kwargs["train_idx"]
    hit_idx = train_idx[df_screen.iloc[train_idx]["y"].values == 1]

    if len(hit_idx) == 0:
        rng = np.random.default_rng(42)
        picks = rng.choice(pool_idx, size=n, replace=False)
        return picks
    X_hits = df_screen.iloc[hit_idx][feature_cols].values.astype(np.int8)
    X_pool = df_screen.iloc[pool_idx][feature_cols].values.astype(np.int8)

    sim_matrix = tanimoto_binary_matrix(X_pool, X_hits)

    max_sim = np.max(sim_matrix, axis=0)

    picks_local = np.argsort(max_sim)[::-1][:n]
    return pool_idx[picks_local]

def similarity_search_cosine(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    df_screen = kwargs["df_screen"]
    feature_cols = kwargs["feature_cols"]
    train_idx = kwargs["train_idx"]

    hit_idx = train_idx[df_screen.iloc[train_idx]["y"].values == 1]

    if len(hit_idx) == 0:
        rng = np.random.default_rng(42)
        n = min(n, len(pool_idx))
        return rng.choice(pool_idx, size=n, replace=False)

    X_hits = df_screen.iloc[hit_idx][feature_cols].values.astype(np.float32)
    X_pool = df_screen.iloc[pool_idx][feature_cols].values.astype(np.float32)

    sim_matrix = cosine_similarity(X_pool, X_hits)
    max_sim = np.max(sim_matrix, axis=1)             
    picks_local = np.argsort(max_sim)[::-1][:n]

    return pool_idx[picks_local]