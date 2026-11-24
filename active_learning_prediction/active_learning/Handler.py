
import numpy as np
from typing import Union, Tuple

class Handler:
    def __init__(self, train_idx: np.ndarray, pool_idx: np.ndarray, y_all: np.ndarray):
        self.train_idx = np.asarray(train_idx, dtype=int).reshape(-1).copy()
        self.pool_idx = np.asarray(pool_idx, dtype=int).reshape(-1).copy()
        self.y_all = y_all

        if len(self.train_idx) != len(np.unique(self.train_idx)):
            raise ValueError('初始训练集存在重复样本索引')
        if len(self.pool_idx) != len(np.unique(self.pool_idx)):
            raise ValueError('初始未标注池存在重复样本索引')
        if np.intersect1d(self.train_idx, self.pool_idx).size:
            raise ValueError('初始训练集与未标注池存在重叠样本索引')

        self.picks = [self.train_idx.copy()]

    def get_idx(self) -> Tuple[np.ndarray, np.ndarray]:

        return self.train_idx, self.pool_idx

    def add(self, picked_idx: Union[np.ndarray, list]):

        picked_idx = np.asarray(picked_idx, dtype=int).reshape(-1)

        if len(picked_idx) != len(np.unique(picked_idx)):
            raise ValueError('本轮获取的样本存在重复索引')

        if not np.isin(picked_idx, self.pool_idx).all():
            raise ValueError('本轮获取的样本不在当前未标注池中，可能存在跨轮次重复获取')

        self.train_idx = np.concatenate([self.train_idx, picked_idx])

        self.pool_idx = self.pool_idx[~np.isin(self.pool_idx, picked_idx)]

        self.picks.append(picked_idx.copy())

    def __call__(self):
        return self.get_idx()
