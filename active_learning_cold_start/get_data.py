import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split




# 测试集所占比例20%，起始集数量64，随机种子42保证可复现
def get_data(file_path,test_size: float = 0.2,n_start: int = 64,random_state: int = 42
             ,feature_cols: list = None,positive_label: int=0,feature_type='int8'):

    # 读取csv文件
    df = pd.read_csv(file_path)

    # 构建二分类标签，label = 0 时，y = 1为正样本，其他情况 y = 0
    df['y'] = df['label'].apply(lambda x: 1 if x == positive_label else 0)
    if feature_cols is not None:
        df[feature_cols] = df[feature_cols].astype(feature_type)

    # 划分测试集与用于主动学习的样本池，保证正负样本比例一致
    df_screen, df_test = train_test_split(df,test_size=test_size,stratify=df['y'],random_state=random_state)

    # 随机数生成器
    rng = np.random.default_rng(seed=random_state)

    # 从主动学习样本池中选出标签列 y
    y_screen = df_screen['y'].values

    # 正样本和负样本的索引
    pos_idx = np.where(y_screen == 1)[0]
    neg_idx = np.where(y_screen == 0)[0]

    # 主动学习样本池中必须同时包含正负样本，如果某一类为空直接报错
    assert len(pos_idx) > 0 and len(neg_idx) > 0, "screen pool 中正或负样本为空，无法初始化"

    # 计算主动学习筛选池中的正样本比例
    pos_ratio = len(pos_idx) / len(y_screen)

    # 初始训练集中正样本的数量
    n_pos_init = int(round(n_start * pos_ratio))

    # 对正样本数量做安全约束，至少是1个，且不超过筛选池中的正样本总数
    n_pos_init = max(1, min(n_pos_init, len(pos_idx)))

    # 负样本数量为初始训练集数量减去正样本数量
    n_neg_init = n_start - n_pos_init

    # 从正负样本中不放回随机抽取
    selected_pos_idx = rng.choice(pos_idx, size=n_pos_init, replace=False)
    selected_neg_idx = rng.choice(neg_idx, size=n_neg_init, replace=False)

    # 合并正负样本索引
    selected_sl_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
    train_idx = selected_sl_idx.copy()

    # 打乱训练集索引顺序
    rng.shuffle(train_idx)

    # 根据训练集索引从样本池中选出训练样本并重置索引
    df_train = df_screen.iloc[train_idx].reset_index(drop=True)

    # 构建未标注池
    pool_idx = np.array([i for i in range(len(df_screen)) if i not in train_idx])
    df_pool = df_screen.iloc[pool_idx].reset_index(drop=True)

    #以下为冷启动部分逻辑
    genes_in_train = set(df_train['Gene A']) | set(df_train['Gene B'])
    mask = (df_pool['Gene A'].isin(genes_in_train) | df_pool['Gene B'].isin(genes_in_train))
    df_pool = df_pool.loc[~mask].reset_index(drop=True)
    # 返回初始训练集、未标注样本池、独立测试集
    return df_train, df_pool, df_test