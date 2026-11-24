
import pandas as pd
import numpy as np

def get_data(file_path, test_size: float = 0.0, n_start: int = 64, random_state: int = 42,
             feature_cols: list = None, positive_label: int = 0, feature_type='int8'):

    if test_size not in (None, 0, 0.0):
        print(f'[INFO] 不再预留测试集：忽略传入的test_size={test_size}')

    dtype_map = {c: feature_type for c in feature_cols} if feature_cols is not None else None
    df = pd.read_csv(file_path, dtype=dtype_map).reset_index(drop=True)

    df['y'] = (df['label'] == positive_label).astype(np.int8)

    df_screen = df
    total = len(df_screen)
    if not 2 <= n_start < total:
        raise ValueError(f'n_start={n_start}不合法；要求2<=n_start<样本总数({total})')

    y = df_screen['y'].to_numpy()
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError('整个数据集中缺少阳性或阴性，无法保证初始训练集包含两个类别')

    rng = np.random.default_rng(random_state)
    train_idx = rng.choice(total, size=n_start, replace=False)
    raw_positive = int(y[train_idx].sum())
    replacement = 'none'

    if raw_positive == 0:
        train_idx[rng.integers(n_start)] = rng.choice(pos_idx)
        replacement = 'add_one_positive'

    elif raw_positive == n_start:
        train_idx[rng.integers(n_start)] = rng.choice(neg_idx)
        replacement = 'add_one_negative'

    rng.shuffle(train_idx)
    if len(np.unique(train_idx)) != n_start: raise AssertionError('初始训练集包含重复样本索引')

    mask = np.ones(total, dtype=bool)
    mask[train_idx] = False
    pool_idx = np.flatnonzero(mask)
    df_train = df_screen.iloc[train_idx].reset_index(drop=True)
    df_pool = df_screen.iloc[pool_idx].reset_index(drop=True)
    final_positive = int(df_train['y'].sum())
    print(f'[INIT] seed={random_state} | n_start={n_start} | raw_positive={raw_positive} | '
          f'final_positive={final_positive} | negative={n_start-final_positive} | '
          f'replacement={replacement} | pool={len(pool_idx)}')

    df_train.attrs['initialization_info'] = {'seed': int(random_state), 'n_start': int(n_start),
        'raw_positive': raw_positive, 'raw_negative': n_start - raw_positive,
        'replacement': replacement, 'final_positive': final_positive,
        'final_negative': n_start - final_positive, 'fully_random_scheme': False}

    df_test = None
    return df_train, df_pool, df_test
