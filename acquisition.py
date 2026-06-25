import numpy as np
from torch import Tensor
import torch
import math
from sklearn.metrics.pairwise import cosine_similarity
# 定义获取函数类
class Acquisition:
    # 策略名与函数的映射表
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

        # 确保 method 拼写无误
        assert method in self.acquisition_method, \
            f"method must be one of {self.acquisition_method.keys()}"

        self.method = method
        self.params = kwargs
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

    # 模型预测结果、未标记池中的样本索引、本轮要选多少个样本
    def acquire(self, logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1) -> np.ndarray:
        # 主动学习轮数加 1
        self.iteration += 1

        # 根据当前策略名称，找到对应函数，执行采样
        selected_idx = self.acquisition_method[self.method](logits_N_K_C=logits_N_K_C,pool_idx=pool_idx,n=n,iteration=self.iteration,
                                                            df_screen=self.df_screen,feature_cols=self.feature_cols,train_idx=self.handler.train_idx,**self.params)

        # 关键修改：所有策略最终返回前，都统一转成一维 int 数组
        selected_idx = to_1d_index_array(selected_idx)

        # 临时检查：确认每轮到底选了几个样本
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

    # 完全随机从未标记样本池中选择样本
    def random_pick(self, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
        picks = self.rng.integers(0, len(pool_idx), n)
        return pool_idx[picks]

# 将 logits 转为预测概率或类别，以及样本不确定性，K 是 ensemble 数量，每个类别的概率，不确定性
def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True):

    # 先计算 ensemble 平均概率
    mean_probs_N_C = torch.mean(torch.exp(logits_N_K_C), dim=1)

    # 计算样本不确定性
    uncertainty = mean_sample_entropy(logits_N_K_C)

    # 根据 return_prob 决定 y_hat 是概率还是类别
    if return_prob:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    # 根据 return_uncertainty 决定返回值
    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat

# 计算 logits 的平均
def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])

# 计算分类熵
def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)

# 对每个模型预测计算熵
def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    # 对 K 个模型求均值
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)
    return entropy_mean_N

# 互信息指标
def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    # 对每个模型的预测熵求平均
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    # 对 K 个模型的预测熵求平均
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)
    # 模型整体的不确定性与各个模型自身的平均不确定性之间的差值
    I = mean_entropy_N - entropy_mean_N
    return I

def to_1d_index_array(x) -> np.ndarray:
    """
    将 acquisition 返回的样本索引统一转成一维 int 数组。
    例如：
        np.int64(3018) -> array([3018])
        [1, 2, 3]     -> array([1, 2, 3])
    """
    if x is None:
        return np.array([], dtype=int)
    arr = np.asarray(x, dtype=int)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.reshape(-1)
    return arr

def safe_concat_picks(picks) -> np.ndarray:
    """
    安全拼接 dynamic 策略中的多组样本索引。
    如果只有一组结果，直接返回；
    如果有多组结果，再 concatenate。
    """
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

# 利用型获取函数
def greedy_exploitation(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    # 对 K 个模型预测值取平均
    mean_probs = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    # 从大到小排序，选取前 n 个
    picks = torch.argsort(mean_probs, descending=True)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

# 探索型获取函数
def greedy_exploration(logits_N_K_C: Tensor, pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    # 使用平均熵作为不确定性指标
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    # 不确定性最大的优先选
    picks = torch.argsort(entropy_mean_N, descending=True)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

# 动态探索——利用平衡，随着迭代的增加，逐渐偏向利用型
def dynamic_exploration(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,lambd: float = 0.95,iteration: int = 0,**kwargs) -> np.ndarray:
    # 控制利用型比例，迭代轮次越多，利用型因子越大
    exploitation_factor = (1 / (lambd ** iteration)) - 1
    # 采用利用型获取函数获取的样本数
    n_exploit = int(round(n * exploitation_factor))
    # 限制安全范围，不能超过未标记样本池样本数
    n_exploit = max(0, min(n_exploit, n, len(pool_idx)))
    # 探索型获取函数获取的样本数
    n_explore = n - n_exploit
    picks = []
    if n_exploit > 0:
        picks.append(greedy_exploitation(logits_N_K_C, pool_idx, n_exploit))
    if n_explore > 0:
        picks.append(greedy_exploration(logits_N_K_C, pool_idx, n_explore))
    return safe_concat_picks(picks)

# 动态探索-互信息，将上面的探索-利用换为探索-互信息
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

# 互信息获取函数
def bald(logits_N_K_C: Tensor,pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    I = mutual_information(logits_N_K_C)
    # 互信息越小的越优先
    picks = torch.argsort(I, descending=False)[:n]
    picks = picks.cpu().numpy()
    selected = pool_idx[picks]
    return to_1d_index_array(selected)

# 定义 Tanimoto 系数
def tanimoto_binary_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    intersection = np.dot(Y, X.T)
    Y_sum = Y.sum(axis=1)[:, None]
    X_sum = X.sum(axis=1)[None, :]
    return intersection / (Y_sum + X_sum - intersection + 1e-10)

# 相似性获取函数
def similarity_search_jaccard(pool_idx: np.ndarray,n: int = 1,**kwargs) -> np.ndarray:
    # 从 kwargs 中拿全局数据
    df_screen = kwargs["df_screen"]
    feature_cols = kwargs["feature_cols"]
    train_idx = kwargs["train_idx"]
    # 找训练集中 y=1 的样本
    hit_idx = train_idx[df_screen.iloc[train_idx]["y"].values == 1]
    # 如果当前没有正样本就退化为随机采样
    if len(hit_idx) == 0:
        rng = np.random.default_rng(42)
        picks = rng.choice(pool_idx, size=n, replace=False)
        return picks
    # 提取二进制 ECFP
    X_hits = df_screen.iloc[hit_idx][feature_cols].values.astype(np.int8)
    X_pool = df_screen.iloc[pool_idx][feature_cols].values.astype(np.int8)
    # 计算 Tanimoto 相似度矩阵
    sim_matrix = tanimoto_binary_matrix(X_pool, X_hits)
    # 对每个 pool 分子取 max similarity
    max_sim = np.max(sim_matrix, axis=0)
    # 选相似度最高的 n 个
    picks_local = np.argsort(max_sim)[::-1][:n]
    return pool_idx[picks_local]
# 浮点特征的 similarity：改为余弦相似度
def similarity_search_cosine(pool_idx: np.ndarray, n: int = 1, **kwargs) -> np.ndarray:
    df_screen = kwargs["df_screen"]
    feature_cols = kwargs["feature_cols"]
    train_idx = kwargs["train_idx"]

    hit_idx = train_idx[df_screen.iloc[train_idx]["y"].values == 1]

    # 没有正样本则退化为随机无放回
    if len(hit_idx) == 0:
        rng = np.random.default_rng(42)
        n = min(n, len(pool_idx))
        return rng.choice(pool_idx, size=n, replace=False)

    X_hits = df_screen.iloc[hit_idx][feature_cols].values.astype(np.float32)
    X_pool = df_screen.iloc[pool_idx][feature_cols].values.astype(np.float32)

    # 用余弦相似度替代二进制 Tanimoto
    sim_matrix = cosine_similarity(X_pool, X_hits)   # shape: (n_pool, n_hits)
    max_sim = np.max(sim_matrix, axis=1)             # 每个 pool 样本与所有 hits 的最大相似度
    picks_local = np.argsort(max_sim)[::-1][:n]

    return pool_idx[picks_local]