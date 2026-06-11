import numpy as np
from typing import Union, Tuple

# Handler 用于封装训练集和未标注池的管理逻辑
class Handler:
    def __init__(self,train_idx: np.ndarray, pool_idx: np.ndarray,y_all: np.ndarray):
        self.train_idx = train_idx.copy()
        self.pool_idx = pool_idx.copy()
        self.y_all = y_all

        # 每一轮主动学习的选择
        self.picks = [self.train_idx.copy()]

    def get_idx(self) -> Tuple[np.ndarray, np.ndarray]:
        # 返回当前训练集索引与未标注池索引
        return self.train_idx, self.pool_idx

    def add(self, picked_idx: Union[np.ndarray, list]):
        # 转换为 NumPy 数组，方便后续拼接
        picked_idx = np.array(picked_idx)

        # 更新训练集，把新选中的样本加入到旧的训练集里
        self.train_idx = np.concatenate([self.train_idx, picked_idx])

        # 更新未标注池，把新选中的样本移出未标注池
        self.pool_idx = np.array([i for i in self.pool_idx if i not in picked_idx])

        # 记录本轮选择
        self.picks.append(picked_idx)

    def __call__(self):
        return self.get_idx()