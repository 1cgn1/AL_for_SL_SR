import os
import re
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path("/data/hdd/MMY/active learning/GEMINI/nsf4sl")

INPUT_FILE = BASE_DIR / "sensitive_Hela_combined_result_NSF4SL_KGemb.csv"
KG_EMB_FILE = BASE_DIR / "kg_TransE_l2_entity.npy"

OUTPUT_DIR = BASE_DIR / "Hela_sensitive_SL_NSF4SL_different_n_start_fullKGmean_benchmarkParams_10repeat"
POSITIVE_LABEL = 0

SEED_LIST = [42, 43, 44, 45, 46,47,48,49,50,51]
N_START_LIST = [16, 32, 64, 128, 256]

METHOD_LIST = [
    "random",
    "exploitation",
    "similarity",
]

TEST_SIZE = 0.2
QUERY_SIZE = 32
ROUNDS = 5
POOL_RATIO = 1.0
FEATURE_TYPE = "float32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PARAMS = {
    "ensemble_size": 10,
    "latent_size": 256,
    "momentum": 0.995,
    "aug_ratio": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "epochs": 100,
    "train_batch_size": 256,
    "predict_batch_size": 4096,
    "device": DEVICE,
}

SIMILARITY_CHUNK_SIZE = 5000
SAVE_FEATURES_IN_SELECTED = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def to_1d_index_array(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(int)

def feature_sort_key(col: str):
    nums = re.findall(r"\d+", str(col))
    if len(nums) == 0:
        return str(col)
    return int(nums[-1])


def get_ab_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    a_cols = [
        c for c in df.columns
        if str(c).endswith("_A") and c not in ["Gene A"]
    ]
    b_cols = [
        c for c in df.columns
        if str(c).endswith("_B") and c not in ["Gene B"]
    ]

    a_cols = sorted(a_cols, key=feature_sort_key)
    b_cols = sorted(b_cols, key=feature_sort_key)

    if len(a_cols) == 0 or len(b_cols) == 0:
        raise ValueError(
            "没有识别到 _A / _B 特征列。请检查输入文件是否包含 "
            "kg_0_A ... kg_399_A 和 kg_0_B ... kg_399_B。"
        )

    if len(a_cols) != len(b_cols):
        raise ValueError(
            f"A/B 特征数量不一致: len(a_cols)={len(a_cols)}, len(b_cols)={len(b_cols)}"
        )

    feature_cols = a_cols + b_cols

    return a_cols, b_cols, feature_cols

def prepare_data(
    file_path: Path,
    positive_label: int,
    test_size: float,
    n_start: int,
    seed: int,
    feature_type: str = "float32",
):
    df = pd.read_csv(file_path)

    if "label" not in df.columns:
        raise ValueError("输入文件必须包含 label 列。")

    df = df.copy()
    df["original_row"] = np.arange(len(df))
    df["y"] = (df["label"] == positive_label).astype(int)

    a_cols, b_cols, feature_cols = get_ab_feature_cols(df)

    df[feature_cols] = df[feature_cols].astype(feature_type)

    if df[feature_cols].isna().sum().sum() > 0:
        raise ValueError("特征列中存在 NaN，请先检查 KG embedding 特征表。")

    if df["y"].nunique() < 2:
        raise ValueError(
            f"二值化后 y 只有一个类别。positive_label={positive_label} 可能设置错误。"
        )

    df_screen, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["y"],
        random_state=seed,
        shuffle=True,
    )

    df_screen = df_screen.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    y_screen = df_screen["y"].values
    pos_idx = np.where(y_screen == 1)[0]
    neg_idx = np.where(y_screen == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("screen pool 中正或负样本为空，无法初始化主动学习。")

    pos_ratio = len(pos_idx) / len(y_screen)

    n_pos_init = int(round(n_start * pos_ratio))
    n_pos_init = max(1, min(n_pos_init, len(pos_idx), n_start - 1))

    n_neg_init = n_start - n_pos_init
    if n_neg_init > len(neg_idx):
        n_neg_init = len(neg_idx)
        n_pos_init = n_start - n_neg_init
        n_pos_init = max(1, min(n_pos_init, len(pos_idx)))

    selected_pos_idx = rng.choice(pos_idx, size=n_pos_init, replace=False)
    selected_neg_idx = rng.choice(neg_idx, size=n_neg_init, replace=False)

    train_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
    rng.shuffle(train_idx)

    train_idx = to_1d_index_array(train_idx)

    train_mask = np.zeros(len(df_screen), dtype=bool)
    train_mask[train_idx] = True
    pool_idx = np.where(~train_mask)[0].astype(int)

    return df_screen, df_test, train_idx, pool_idx, a_cols, b_cols, feature_cols



def load_full_kg_embedding_mean(expected_dim: int) -> np.ndarray:
    if not KG_EMB_FILE.exists():
        raise FileNotFoundError(
            f"KG embedding file not found: {KG_EMB_FILE}\n"
            "请确认 kg_TransE_l2_entity.npy 的真实路径，并修改 KG_EMB_FILE。"
        )

    kgemb_data = np.load(KG_EMB_FILE, mmap_mode="r")
    kgemb_mean = np.asarray(np.mean(kgemb_data, axis=0), dtype=np.float32)

    if kgemb_mean.shape[0] != expected_dim:
        raise ValueError(
            "KG embedding mean 维度与当前 A/B 特征维度不一致："
            f"kgemb_mean_dim={kgemb_mean.shape[0]}, expected_dim={expected_dim}. "
            "请检查 KG_EMB_FILE 是否与 sensitive_Hela_combined_result_NSF4SL_KGemb.csv "
            "来自同一套 KG embedding。"
        )

    print(
        f"[KG mean] Using full KG embedding mean from {KG_EMB_FILE}; "
        f"kgemb_shape={kgemb_data.shape}; mean_dim={kgemb_mean.shape[0]}"
    )

    return kgemb_mean

class Handler:
    def __init__(self, train_idx: np.ndarray, pool_idx: np.ndarray, y_all: np.ndarray):
        self.train_idx = to_1d_index_array(train_idx)
        self.pool_idx = to_1d_index_array(pool_idx)
        self.y_all = np.asarray(y_all)
        self.picks = [self.train_idx.copy()]

    def get_idx(self):
        return self.train_idx, self.pool_idx

    def add(self, picked_idx):
        picked_idx = to_1d_index_array(picked_idx)

        if len(picked_idx) == 0:
            return

        picked_idx = np.unique(picked_idx)

        self.train_idx = np.concatenate([self.train_idx, picked_idx]).astype(int)

        picked_set = set(picked_idx.tolist())
        self.pool_idx = np.array(
            [i for i in self.pool_idx if int(i) not in picked_set],
            dtype=int,
        )

        self.picks.append(picked_idx.copy())

def xavier_init(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class MLP(nn.Module):
    def __init__(self, input_size, projection_size, hid_size1=512, hid_size2=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size1),
            nn.BatchNorm1d(hid_size1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size1, hid_size2),
            nn.BatchNorm1d(hid_size2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid_size2, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class NSF4SLNet(nn.Module):
    def __init__(self, input_size, args):
        super(NSF4SLNet, self).__init__()
        self.latent_size = args.latent_size
        self.momentum = args.momentum
        self.input_size = input_size

        self.online_encoder = MLP(input_size, self.latent_size)
        self.target_encoder = MLP(input_size, self.latent_size)

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self.online_encoder.apply(xavier_init)
        self.predictor.apply(xavier_init)
        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_t.data = param_t.data * self.momentum + param_o.data * (1.0 - self.momentum)

    def forward(self, inputs):
        g1, g2, g1_aug, g2_aug = inputs[0], inputs[1], inputs[2], inputs[3]

        g1_online = self.predictor(self.online_encoder(g1_aug))
        g1_target = self.target_encoder(g1)

        g2_online = self.predictor(self.online_encoder(g2_aug))
        g2_target = self.target_encoder(g2)

        return g1_online, g1_target, g2_online, g2_target

    @torch.no_grad()
    def get_embedding(self, inputs):
        g_online = self.online_encoder(inputs.float())
        return self.predictor(g_online), g_online

    def get_loss(self, output):
        u_online, u_target, i_online, i_target = output

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    @torch.no_grad()
    def score_pair(self, x_a, x_b):
        x_a = x_a.float()
        x_b = x_b.float()

        a_online = self.predictor(self.online_encoder(x_a))
        b_online = self.predictor(self.online_encoder(x_b))

        a_target = self.target_encoder(x_a)
        b_target = self.target_encoder(x_b)

        a_online = F.normalize(a_online, dim=-1)
        b_online = F.normalize(b_online, dim=-1)
        a_target = F.normalize(a_target, dim=-1)
        b_target = F.normalize(b_target, dim=-1)

        score_ab = (a_online * b_target).sum(dim=-1)
        score_ba = (b_online * a_target).sum(dim=-1)

        return 0.5 * (score_ab + score_ba)


class NSF4SLPairDataset(Dataset):
    def __init__(
        self,
        x_a: np.ndarray,
        x_b: np.ndarray,
        feature_mean: np.ndarray,
        aug_ratio: float,
        seed: int,
    ):
        self.x_a = np.asarray(x_a, dtype=np.float32)
        self.x_b = np.asarray(x_b, dtype=np.float32)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.aug_ratio = float(aug_ratio)
        self.rng = np.random.default_rng(seed)

        if self.x_a.shape != self.x_b.shape:
            raise ValueError(f"x_a 和 x_b shape 不一致: {self.x_a.shape}, {self.x_b.shape}")

        if self.feature_mean.shape[0] != self.x_a.shape[1]:
            raise ValueError(
                f"feature_mean 维度与输入不一致: {self.feature_mean.shape[0]} vs {self.x_a.shape[1]}"
            )

    def __len__(self):
        return self.x_a.shape[0]

    def _augment(self, x):
        x_aug = x.copy()
        dim = x_aug.shape[0]

        if self.aug_ratio <= 0:
            return x_aug

        n_mask = int(round(dim * self.aug_ratio))
        n_mask = max(1, min(n_mask, dim))

        mask_ids = self.rng.choice(dim, size=n_mask, replace=False)
        x_aug[mask_ids] = self.feature_mean[mask_ids]

        return x_aug

    def __getitem__(self, idx):
        g1 = self.x_a[idx]
        g2 = self.x_b[idx]

        g1_aug = self._augment(g1)
        g2_aug = self._augment(g2)

        return (
            torch.tensor(g1, dtype=torch.float32),
            torch.tensor(g2, dtype=torch.float32),
            torch.tensor(g1_aug, dtype=torch.float32),
            torch.tensor(g2_aug, dtype=torch.float32),
        )


class NSF4SLRankingEnsemble:
    def __init__(
        self,
        input_dim: int,
        feature_mean: np.ndarray,
        ensemble_size: int = 10,
        latent_size: int = 256,
        momentum: float = 0.9,
        aug_ratio: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        train_batch_size: int = 64,
        predict_batch_size: int = 4096,
        seed: int = 42,
        device: str = "cuda",
    ):
        self.input_dim = int(input_dim)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)

        self.ensemble_size = int(ensemble_size)
        self.latent_size = int(latent_size)
        self.momentum = float(momentum)
        self.aug_ratio = float(aug_ratio)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.train_batch_size = int(train_batch_size)
        self.predict_batch_size = int(predict_batch_size)
        self.seed = int(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(0, 1000000, self.ensemble_size)

        args = SimpleNamespace(
            latent_size=self.latent_size,
            momentum=self.momentum,
        )

        self.models = []
        self.optimizers = []

        for s in self.seeds:
            torch.manual_seed(int(s))
            if torch.cuda.is_available():
                torch.cuda.manual_seed(int(s))

            net = NSF4SLNet(self.input_dim, args).to(self.device)
            opt = torch.optim.Adam(
                net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            self.models.append(net)
            self.optimizers.append(opt)

    def train(self, x_a_all: np.ndarray, x_b_all: np.ndarray, y: np.ndarray):
        y = np.asarray(y).astype(int)
        pos_mask = (y == 1)
        n_pos = int(pos_mask.sum())

        if n_pos < 2:
            print(
                "[NSF4SL warning] 当前训练集阳性样本少于 2 个，"
                "本轮 exploitation 跳过对比学习训练，使用初始化模型打分。"
            )
            return

        x_a_pos = np.asarray(x_a_all[pos_mask], dtype=np.float32)
        x_b_pos = np.asarray(x_b_all[pos_mask], dtype=np.float32)

        dataset = NSF4SLPairDataset(
            x_a=x_a_pos,
            x_b=x_b_pos,
            feature_mean=self.feature_mean,
            aug_ratio=self.aug_ratio,
            seed=self.seed,
        )

        batch_size = min(self.train_batch_size, len(dataset))
        if batch_size < 2:
            print("[NSF4SL warning] batch_size < 2，跳过训练。")
            return

        drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=0,
        )

        for model, opt in zip(self.models, self.optimizers):
            model.train()

            for _ in range(self.epochs):
                for g1, g2, g1_aug, g2_aug in loader:
                    if g1.shape[0] <= 1:
                        continue

                    g1 = g1.to(self.device)
                    g2 = g2.to(self.device)
                    g1_aug = g1_aug.to(self.device)
                    g2_aug = g2_aug.to(self.device)

                    output = model((g1, g2, g1_aug, g2_aug))
                    loss = model.get_loss(output)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    model._update_target()

    @torch.no_grad()
    def predict_scores(self, x_a: np.ndarray, x_b: np.ndarray) -> torch.Tensor:
        x_a = np.asarray(x_a, dtype=np.float32)
        x_b = np.asarray(x_b, dtype=np.float32)

        n = x_a.shape[0]
        all_model_scores = []

        for model in self.models:
            model.eval()
            model_scores = []

            for start in range(0, n, self.predict_batch_size):
                end = min(start + self.predict_batch_size, n)

                xa_batch = torch.tensor(
                    x_a[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                xb_batch = torch.tensor(
                    x_b[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                score = model.score_pair(xa_batch, xb_batch)
                model_scores.append(score.cpu())

            model_scores = torch.cat(model_scores, dim=0)
            all_model_scores.append(model_scores.unsqueeze(1))

        scores_n_k = torch.cat(all_model_scores, dim=1)

        return scores_n_k

def acquire_random(pool_idx: np.ndarray, n: int, rng: np.random.Generator):
    n = min(n, len(pool_idx))
    selected = rng.choice(pool_idx, size=n, replace=False)
    selected = to_1d_index_array(selected)
    selected_score = np.full(len(selected), np.nan, dtype=np.float32)
    return selected, selected_score


def acquire_exploitation_by_score(
    pool_idx: np.ndarray,
    scores_n_k: torch.Tensor,
    n: int,
):
    n = min(n, len(pool_idx))

    mean_score = torch.mean(scores_n_k, dim=1).cpu().numpy()

    picks_local = np.argsort(mean_score)[::-1][:n]

    selected = to_1d_index_array(pool_idx[picks_local])
    selected_score = mean_score[picks_local]

    return selected, selected_score


def acquire_similarity_float(
    df_screen: pd.DataFrame,
    train_idx: np.ndarray,
    pool_idx: np.ndarray,
    pair_feature_cols: List[str],
    n: int,
    rng: np.random.Generator,
):
    n = min(n, len(pool_idx))

    train_idx = to_1d_index_array(train_idx)
    pool_idx = to_1d_index_array(pool_idx)

    hit_idx = train_idx[df_screen.iloc[train_idx]["y"].values == 1]

    if len(hit_idx) == 0:
        selected = rng.choice(pool_idx, size=n, replace=False)
        selected = to_1d_index_array(selected)
        selected_score = np.full(len(selected), np.nan, dtype=np.float32)
        return selected, selected_score

    x_hits = df_screen.iloc[hit_idx][pair_feature_cols].values.astype(np.float32)

    max_sim_all = np.full(len(pool_idx), -np.inf, dtype=np.float32)

    for start in range(0, len(pool_idx), SIMILARITY_CHUNK_SIZE):
        end = min(start + SIMILARITY_CHUNK_SIZE, len(pool_idx))
        chunk_idx = pool_idx[start:end]

        x_pool = df_screen.iloc[chunk_idx][pair_feature_cols].values.astype(np.float32)
        sim_matrix = cosine_similarity(x_pool, x_hits)
        max_sim = np.max(sim_matrix, axis=1).astype(np.float32)

        max_sim_all[start:end] = max_sim

    picks_local = np.argsort(max_sim_all)[::-1][:n]

    selected = to_1d_index_array(pool_idx[picks_local])
    selected_score = max_sim_all[picks_local]

    return selected, selected_score

class NSF4SLActiveLearningExperiment:
    def __init__(
        self,
        file_path: Path,
        output_dir: Path,
        method: str,
        seed: int,
        n_start: int,
        positive_label: int,
        test_size: float,
        query_size: int,
        rounds: int,
        pool_ratio: float,
        feature_type: str,
        model_params: Dict,
    ):
        self.file_path = Path(file_path)
        self.output_dir = Path(output_dir)

        self.method = method
        self.seed = int(seed)
        self.n_start = int(n_start)
        self.positive_label = int(positive_label)
        self.test_size = float(test_size)
        self.query_size = int(query_size)
        self.rounds = int(rounds)
        self.pool_ratio = float(pool_ratio)
        self.feature_type = feature_type
        self.model_params = dict(model_params)

        if self.method not in ["random", "exploitation", "similarity"]:
            raise ValueError(
                "method must be one of: random, exploitation, similarity"
            )

        self.df_screen = None
        self.df_test = None
        self.handler = None

        self.a_cols = None
        self.b_cols = None
        self.pair_feature_cols = None
        self.feature_mean = None

        self.history = []
        self.selected_records = []

        self.initial_positive = 0
        self.initial_train_size = 0
        self.pos_neg_ratio = None

    def prepare(self):
        (
            df_screen,
            df_test,
            train_idx,
            pool_idx,
            a_cols,
            b_cols,
            feature_cols,
        ) = prepare_data(
            file_path=self.file_path,
            positive_label=self.positive_label,
            test_size=self.test_size,
            n_start=self.n_start,
            seed=self.seed,
            feature_type=self.feature_type,
        )

        self.df_screen = df_screen
        self.df_test = df_test
        self.a_cols = a_cols
        self.b_cols = b_cols
        self.pair_feature_cols = feature_cols

        y_all = self.df_screen["y"].values
        self.handler = Handler(train_idx=train_idx, pool_idx=pool_idx, y_all=y_all)

        self.feature_mean = load_full_kg_embedding_mean(
            expected_dim=len(self.a_cols)
        )

        n_pos = int(self.df_screen.iloc[train_idx]["y"].sum())
        n_neg = int(len(train_idx) - n_pos)

        self.initial_positive = n_pos
        self.initial_train_size = len(train_idx)
        self.pos_neg_ratio = f"{n_pos}:{n_neg}"

        self._record_history(
            cycle=0,
            selected_idx=np.array([], dtype=int),
            selected_score=np.array([], dtype=np.float32),
        )

    def _build_model(self):
        params = dict(self.model_params)
        params["seed"] = self.seed
        params["input_dim"] = len(self.a_cols)
        params["feature_mean"] = self.feature_mean

        return NSF4SLRankingEnsemble(**params)

    def _get_pool_subset(self, pool_idx: np.ndarray, cycle: int):
        pool_idx = to_1d_index_array(pool_idx)

        if self.pool_ratio >= 1.0:
            return pool_idx

        n_sub = int(round(len(pool_idx) * self.pool_ratio))
        n_sub = max(1, min(n_sub, len(pool_idx)))

        df_pool_current = self.df_screen.iloc[pool_idx]

        try:
            df_subset, _ = train_test_split(
                df_pool_current,
                train_size=n_sub,
                stratify=df_pool_current["y"],
                shuffle=True,
                random_state=self.seed + cycle,
            )
        except ValueError:
            df_subset, _ = train_test_split(
                df_pool_current,
                train_size=n_sub,
                shuffle=True,
                random_state=self.seed + cycle,
            )

        return to_1d_index_array(df_subset.index.values)

    def _get_xa_xb(self, idx: np.ndarray):
        idx = to_1d_index_array(idx)
        x_a = self.df_screen.iloc[idx][self.a_cols].values.astype(np.float32)
        x_b = self.df_screen.iloc[idx][self.b_cols].values.astype(np.float32)
        return x_a, x_b

    def _acquire(self, cycle: int):
        rng = np.random.default_rng(self.seed + cycle * 1009)

        train_idx, pool_idx = self.handler.get_idx()
        current_batch = min(self.query_size, len(pool_idx))

        if current_batch <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        pool_idx_one_cycle = self._get_pool_subset(pool_idx, cycle=cycle)

        if self.method == "random":
            return acquire_random(
                pool_idx=pool_idx_one_cycle,
                n=current_batch,
                rng=rng,
            )

        if self.method == "similarity":
            return acquire_similarity_float(
                df_screen=self.df_screen,
                train_idx=train_idx,
                pool_idx=pool_idx_one_cycle,
                pair_feature_cols=self.pair_feature_cols,
                n=current_batch,
                rng=rng,
            )

        if self.method == "exploitation":
            x_train_a, x_train_b = self._get_xa_xb(train_idx)
            y_train = self.df_screen.iloc[train_idx]["y"].values.astype(int)

            x_pool_a, x_pool_b = self._get_xa_xb(pool_idx_one_cycle)

            model = self._build_model()
            model.train(x_train_a, x_train_b, y_train)

            scores_n_k = model.predict_scores(x_pool_a, x_pool_b)

            return acquire_exploitation_by_score(
                pool_idx=pool_idx_one_cycle,
                scores_n_k=scores_n_k,
                n=current_batch,
            )

        raise ValueError(f"Unknown method: {self.method}")

    def _record_selected_samples(
        self,
        cycle: int,
        selected_idx: np.ndarray,
        selected_score: np.ndarray,
    ):
        selected_idx = to_1d_index_array(selected_idx)

        if len(selected_idx) == 0:
            return

        selected_score = np.asarray(selected_score)
        if len(selected_score) != len(selected_idx):
            selected_score = np.full(len(selected_idx), np.nan)

        score_map = {
            int(idx): float(score) if not pd.isna(score) else np.nan
            for idx, score in zip(selected_idx, selected_score)
        }

        base_cols = ["label", "y", "original_row"]
        if "Gene A" in self.df_screen.columns:
            base_cols.insert(0, "Gene A")
        if "Gene B" in self.df_screen.columns:
            insert_pos = 1 if "Gene A" in self.df_screen.columns else 0
            base_cols.insert(insert_pos, "Gene B")

        if SAVE_FEATURES_IN_SELECTED:
            base_cols = base_cols + self.pair_feature_cols

        for idx in selected_idx:
            idx = int(idx)
            row = self.df_screen.iloc[idx]

            record = {
                "dataset": self.file_path.name,
                "method": self.method,
                "seed": self.seed,
                "n_start": self.n_start,
                "cycle": cycle,
                "screen_index": idx,
                "acquisition_score": score_map.get(idx, np.nan),
            }

            for c in base_cols:
                if c in self.df_screen.columns:
                    record[c] = row[c]

            self.selected_records.append(record)

    def _record_history(
        self,
        cycle: int,
        selected_idx: np.ndarray,
        selected_score: np.ndarray,
    ):
        train_idx, pool_idx = self.handler.get_idx()

        selected_idx = to_1d_index_array(selected_idx)
        selected_count = len(selected_idx)

        if selected_count > 0:
            selected_positive = int(self.df_screen.iloc[selected_idx]["y"].sum())
        else:
            selected_positive = 0

        n_positive = int(self.df_screen.iloc[train_idx]["y"].sum())
        n_new_positive = int(n_positive - self.initial_positive)

        screening_budget = int(len(train_idx) - self.initial_train_size)

        if screening_budget > 0:
            positive_discovery_rate = n_new_positive / screening_budget
        else:
            positive_discovery_rate = 0.0

        if len(selected_score) > 0 and not np.all(pd.isna(selected_score)):
            mean_acq_score = float(np.nanmean(selected_score))
        else:
            mean_acq_score = np.nan

        self.history.append(
            {
                "dataset": self.file_path.name,
                "method": self.method,
                "seed": self.seed,
                "n_start": self.n_start,
                "cycle": cycle,
                "train_size": int(len(train_idx)),
                "pool_size": int(len(pool_idx)),
                "selected_count": int(selected_count),
                "selected_positive": int(selected_positive),
                "n_positive": int(n_positive),
                "initial_positive": int(self.initial_positive),
                "n_new_positive": int(n_new_positive),
                "screening_budget": int(screening_budget),
                "positive_discovery_rate": float(positive_discovery_rate),
                "pos_neg_ratio": self.pos_neg_ratio,
                "mean_acquisition_score": mean_acq_score,
            }
        )

    def run(self):
        self.prepare()

        print(
            f"[START] method={self.method} | seed={self.seed} | "
            f"n_start={self.n_start} | positive_label={self.positive_label}"
        )

        for cycle in range(1, self.rounds + 1):
            train_idx, pool_idx = self.handler.get_idx()

            if len(pool_idx) == 0:
                print("Pool empty, stopping.")
                break

            selected_idx, selected_score = self._acquire(cycle=cycle)

            if len(selected_idx) == 0:
                print("No sample selected, stopping.")
                break

            self.handler.add(selected_idx)

            self._record_selected_samples(
                cycle=cycle,
                selected_idx=selected_idx,
                selected_score=selected_score,
            )

            self._record_history(
                cycle=cycle,
                selected_idx=selected_idx,
                selected_score=selected_score,
            )

            current_record = self.history[-1]

            print(
                f"Cycle {cycle:02d} | "
                f"Train={current_record['train_size']:5d} | "
                f"Pool={current_record['pool_size']:5d} | "
                f"Positives={current_record['n_positive']:5d} | "
                f"NewPos={current_record['n_new_positive']:5d} | "
                f"SelectedPos={current_record['selected_positive']:3d}"
            )

        history_df = pd.DataFrame(self.history)
        selected_df = pd.DataFrame(self.selected_records)

        return history_df, selected_df

def summarize_mean_sd_se(df: pd.DataFrame):
    group_cols = [
        "dataset",
        "method",
        "n_start",
        "cycle",
    ]

    numeric_cols = [
        "train_size",
        "pool_size",
        "selected_count",
        "selected_positive",
        "n_positive",
        "initial_positive",
        "n_new_positive",
        "screening_budget",
        "positive_discovery_rate",
        "mean_acquisition_score",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else int(len(sub))

        for col in numeric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() >= 2 else 0.0
            row[f"{col}_se"] = (
                row[f"{col}_sd"] / math.sqrt(vals.notna().sum())
                if vals.notna().sum() >= 2
                else 0.0
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(group_cols).reset_index(drop=True)

    return out

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)

    print("=" * 80)
    print("NSF4SL active learning")
    print(f"INPUT_FILE = {INPUT_FILE}")
    print(f"KG_EMB_FILE = {KG_EMB_FILE}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"POSITIVE_LABEL = {POSITIVE_LABEL}")
    print(f"N_START_LIST = {N_START_LIST}")
    print(f"METHOD_LIST = {METHOD_LIST}")
    print(f"SEED_LIST = {SEED_LIST}")
    print(f"DEVICE = {DEVICE}")
    print("=" * 80)

    df_check = pd.read_csv(INPUT_FILE)
    a_cols, b_cols, feature_cols = get_ab_feature_cols(df_check)

    print("[CHECK] input shape:", df_check.shape)
    print("[CHECK] label distribution:")
    print(df_check["label"].value_counts().sort_index())
    print("[CHECK] A feature count:", len(a_cols))
    print("[CHECK] B feature count:", len(b_cols))
    print("[CHECK] NaN in features:", int(df_check[feature_cols].isna().sum().sum()))

    if int(df_check[feature_cols].isna().sum().sum()) > 0:
        raise ValueError("输入特征中存在 NaN，请先处理。")

    all_history = []
    all_selected = []

    for n_start in N_START_LIST:
        for method in METHOD_LIST:
            for seed in SEED_LIST:
                set_seed(seed)

                exp = NSF4SLActiveLearningExperiment(
                    file_path=INPUT_FILE,
                    output_dir=OUTPUT_DIR,
                    method=method,
                    seed=seed,
                    n_start=n_start,
                    positive_label=POSITIVE_LABEL,
                    test_size=TEST_SIZE,
                    query_size=QUERY_SIZE,
                    rounds=ROUNDS,
                    pool_ratio=POOL_RATIO,
                    feature_type=FEATURE_TYPE,
                    model_params=MODEL_PARAMS,
                )

                history_df, selected_df = exp.run()

                all_history.append(history_df)

                if len(selected_df) > 0:
                    all_selected.append(selected_df)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    results_all = pd.concat(all_history, axis=0, ignore_index=True)

    results_all_path = OUTPUT_DIR / "active_learning_results_all_seeds.csv"
    results_all.to_csv(results_all_path, index=False, encoding="utf-8-sig")

    results_summary = summarize_mean_sd_se(results_all)
    results_summary_path = OUTPUT_DIR / "active_learning_results_mean_sd_se.csv"
    results_summary.to_csv(results_summary_path, index=False, encoding="utf-8-sig")

    if len(all_selected) > 0:
        selected_all = pd.concat(all_selected, axis=0, ignore_index=True)
    else:
        selected_all = pd.DataFrame()

    selected_path = OUTPUT_DIR / "selected_samples_all_seeds.csv"
    selected_all.to_csv(selected_path, index=False, encoding="utf-8-sig")

    config = {
        "base_dir": str(BASE_DIR),
        "input_file": str(INPUT_FILE),
        "kg_emb_file": str(KG_EMB_FILE),
        "feature_mean_source": "full_kg_TransE_l2_entity_mean",
        "output_dir": str(OUTPUT_DIR),
        "positive_label": POSITIVE_LABEL,
        "seed_list": SEED_LIST,
        "n_start_list": N_START_LIST,
        "method_list": METHOD_LIST,
        "test_size": TEST_SIZE,
        "query_size": QUERY_SIZE,
        "rounds": ROUNDS,
        "pool_ratio": POOL_RATIO,
        "feature_type": FEATURE_TYPE,
        "device": DEVICE,
        "model_params": MODEL_PARAMS,
        "input_shape": list(df_check.shape),
        "a_feature_count": len(a_cols),
        "b_feature_count": len(b_cols),
        "label_distribution": {
            str(k): int(v)
            for k, v in df_check["label"].value_counts().sort_index().items()
        },
    }

    config_path = OUTPUT_DIR / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("[DONE] NSF4SL active learning finished.")
    print(f"1. {results_all_path}")
    print(f"2. {results_summary_path}")
    print(f"3. {selected_path}")
    print(f"4. {config_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
