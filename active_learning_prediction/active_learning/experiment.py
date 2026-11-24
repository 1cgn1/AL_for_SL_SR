
import os
import re
import numpy as np
import pandas as pd
from active_learning.get_data import get_data
from active_learning.Handler import Handler
from active_learning.acquisition import Acquisition
import active_learning.model as model

DEFAULT_OUTPUT_DIR = '/data/hdd/MMY/active learning/GEMINI_return/run_model/acquisition_details'

class Experiment:

    def __init__(self, file_path: str, feature_cols: list, test_size: float = 0.0, n_start: int = 64,
                 batch_size: int = 32, max_screen_size: int = None, acquisition_method: str = 'random',
                 model_type='rf', model_params=None, seed: int = 42, n_jobs=10, learning_rate: float = 0.01,
                 drop_out_p: float = 0.2, epochs: int = 5, positive_label: int = 0, feature_type='int8',
                 pool_ratio: float = 1.0, output_dir: str = DEFAULT_OUTPUT_DIR, **params):
        if test_size not in (None, 0, 0.0): raise ValueError('返修实验取消20%测试集，请将test_size设为0')
        if pool_ratio != 1: raise ValueError('返修实验采用完整候选池，请将pool_ratio设为1')
        if n_start < 2 or batch_size < 1: raise ValueError('n_start必须>=2，batch_size必须>=1')
        self.file_path, self.feature_cols = file_path, list(feature_cols)
        self.test_size, self.n_start, self.batch_size = 0.0, int(n_start), int(batch_size)
        self.max_screen_size, self.acquisition_method, self.model_type = max_screen_size, acquisition_method, model_type
        self.model_params = {} if model_params is None else dict(model_params)
        self.seed, self.n_jobs, self.input_dim = int(seed), n_jobs, len(self.feature_cols)
        self.learning_rate, self.drop_out_p, self.epochs = learning_rate, drop_out_p, epochs
        self.positive_label, self.feature_type, self.pool_ratio = positive_label, feature_type, 1.0
        self.output_dir, self.params = output_dir, dict(params)
        self.df_screen = self.df_train = self.df_pool = self.df_test = None
        self.handler = self.model = self.acquisition = None
        self.acquired_records, self.initialization_info = [], {}
        self.history = {k: [] for k in ['cycle', 'train_size', 'pool_size', 'n_positive', 'n_start', 'pos_neg_ratio',
                                         'current_pos_neg_ratio', 'n_new_positive', 'n_positive_new_cumulative',
                                         'n_exploit', 'n_explore', 'n_unique_gene_pairs']}
        self.input_name = os.path.splitext(os.path.basename(file_path))[0]

        match = re.fullmatch(r'(?P<cell>.+?)_(?P<label_type>sensitive|strong)_(?P<pct>\d+)pct_(?P<feature>function|protein)', self.input_name, flags=re.I)
        self.cell_line = match.group('cell') if match else 'unknown'
        self.label_type = match.group('label_type').lower() if match else 'unknown'
        self.threshold_pct = int(match.group('pct')) if match else None
        self.feature_name = match.group('feature').lower() if match else 'unknown'
        self.task = 'SL' if positive_label == 0 else 'SR' if positive_label == 1 else f'label{positive_label}'

    def _build_model(self):

        return model.build_model(model_type=self.model_type, seed=self.seed, params=self.model_params,
                                 n_jobs=self.n_jobs, input_dim=self.input_dim, learning_rate=self.learning_rate,
                                 drop_out_p=self.drop_out_p, epochs=self.epochs)

    def prepare_data(self):

        df_train, df_pool, _ = get_data(file_path=self.file_path, test_size=0.0, n_start=self.n_start,
                                        random_state=self.seed, feature_cols=self.feature_cols,
                                        positive_label=self.positive_label, feature_type=self.feature_type)

        self.initialization_info = dict(df_train.attrs.get('initialization_info', {}))
        self.df_train, self.df_pool, self.df_test = df_train, df_pool, None
        self.df_screen = pd.concat([df_train, df_pool], ignore_index=True)
        if not {'Gene A', 'Gene B', 'label', 'y'}.issubset(self.df_screen.columns):
            raise KeyError('输入CSV需要Gene A、Gene B、label列，get_data需要生成二分类y列')

        gene_a = self.df_screen['Gene A'].astype(str).str.strip()
        gene_b = self.df_screen['Gene B'].astype(str).str.strip()
        if (gene_a == '').any() or (gene_b == '').any() or (gene_a == gene_b).any():
            raise ValueError('Gene A/Gene B包含空值或自配对')
        pair_keys = list(zip(np.minimum(gene_a, gene_b), np.maximum(gene_a, gene_b)))
        if len(pair_keys) != len(set(pair_keys)): raise ValueError('存在重复无序基因对，无法保证阳性hit唯一计数')

        train_idx = np.arange(len(df_train), dtype=int)
        pool_idx = np.arange(len(df_train), len(self.df_screen), dtype=int)
        self.handler = Handler(train_idx, pool_idx, self.df_screen['y'].to_numpy())
        self.acquisition = Acquisition(method=self.acquisition_method, seed=self.seed)
        self.acquisition.df_screen = self.df_screen
        self.acquisition.feature_cols = self.feature_cols
        self.acquisition.handler = self.handler
        self.model = self._build_model()
        if self.max_screen_size is None: self.max_screen_size = len(self.df_screen)
        if not self.n_start <= self.max_screen_size <= len(self.df_screen):
            raise ValueError('max_screen_size必须位于n_start与完整样本数之间')
        n_pos = int(df_train['y'].sum()); n_neg = len(df_train) - n_pos
        self.pos_neg_ratio = f'{n_pos}:{n_neg}'

        replacement = self.initialization_info.get('replacement', 'unknown')
        raw_pos = self.initialization_info.get('raw_positive', 'unknown')
        unchanged = replacement == 'none' if replacement != 'unknown' else 'unknown'
        print(f'[INIT CHECK] seed={self.seed} | method=random_without_replacement_then_min_class_constraint | '
              f'raw_positive={raw_pos} | replacement={replacement} | unchanged_random_draw={unchanged} | '
              f'fully_random_scheme=NO | final_positive={n_pos} | final_negative={n_neg}', flush=True)
        self._record_acquired(0, train_idx, ['initial'] * len(train_idx), train_idx_before=np.empty(0, dtype=int),
                              train_size_before=0, pool_size_before=len(self.df_screen))

    def _record_acquired(self, cycle, picked_idx, sources, train_idx_before, train_size_before, pool_size_before):

        rows = self.df_screen.iloc[picked_idx]
        if len(rows) != len(sources): raise AssertionError('获取索引与阶段来源长度不一致')
        seen = set()
        if cycle > 0:
            known = self.df_screen.iloc[train_idx_before][['Gene A', 'Gene B']].to_numpy()
            seen = set(known.reshape(-1).tolist())
        for idx, (_, row), source in zip(picked_idx, rows.iterrows(), sources):
            a, b = str(row['Gene A']), str(row['Gene B'])
            self.acquired_records.append({'input_file': os.path.basename(self.file_path), 'cell_line': self.cell_line,
                'feature': self.feature_name, 'label_type': self.label_type, 'threshold_pct': self.threshold_pct,
                'task': self.task, 'model': self.model_type, 'method': self.acquisition_method, 'seed': self.seed,
                'n_start': self.n_start, 'cycle': cycle, 'sample_idx': int(idx), 'Gene A': a, 'Gene B': b,
                'label': row['label'], 'y': int(row['y']), 'is_positive': int(row['y'] == 1),
                'acquisition_source': source, 'seen_gene_count_before': int(a in seen) + int(b in seen) if cycle else np.nan,
                'train_size_before': train_size_before, 'pool_size_before': pool_size_before})

    def _record_history(self, cycle, n_new_positive, n_exploit=0, n_explore=0):
        train_idx, pool_idx = self.handler.get_idx()
        y_train = self.df_screen.iloc[train_idx]['y'].to_numpy()
        n_pos = int(np.sum(y_train == 1)); n_neg = int(np.sum(y_train == 0))
        initial_pos = int(self.df_train['y'].sum())
        vals = {'cycle': cycle, 'train_size': len(train_idx), 'pool_size': len(pool_idx),
                'n_positive': n_pos, 'n_start': self.n_start, 'pos_neg_ratio': self.pos_neg_ratio,
                'current_pos_neg_ratio': f'{n_pos}:{n_neg}', 'n_new_positive': int(n_new_positive),
                'n_positive_new_cumulative': n_pos - initial_pos, 'n_exploit': int(n_exploit),
                'n_explore': int(n_explore), 'n_unique_gene_pairs': len(train_idx)}
        for key, value in vals.items(): self.history[key].append(value)

    def run_cycle(self, cycle: int):
        train_idx, pool_idx = self.handler.get_idx()
        if len(pool_idx) == 0 or len(train_idx) >= self.max_screen_size: return False

        n = min(self.batch_size, len(pool_idx), self.max_screen_size - len(train_idx))
        if n <= 0: return False
        X_train = self.df_screen.iloc[train_idx][self.feature_cols].to_numpy(copy=False)
        y_train = self.df_screen.iloc[train_idx]['y'].to_numpy()
        X_pool = self.df_screen.iloc[pool_idx][self.feature_cols].to_numpy(copy=False)

        print(f'[CYCLE {cycle}] train={len(train_idx)} | full_pool_scored={len(pool_idx)} | batch={n}', flush=True)
        self.model.train(X_train, y_train)

        logits = self.model.predict(X_pool)
        picked_idx = self.acquisition(logits_N_K_C=logits, pool_idx=pool_idx, n=n)
        if len(picked_idx) == 0: return False
        sources = self.acquisition.last_selected_sources
        if len(sources) != len(picked_idx): raise AssertionError('acquisition未记录逐样本获取来源，请使用两项修改版获取策略')
        details = self.acquisition.last_acquisition_details

        self.handler.add(picked_idx)
        self._record_acquired(cycle, picked_idx, sources, train_idx_before=train_idx,
                              train_size_before=len(train_idx), pool_size_before=len(pool_idx))
        n_new_positive = int(self.df_screen.iloc[picked_idx]['y'].sum())
        self._record_history(cycle, n_new_positive, details.get('n_exploit', 0), details.get('n_explore', 0))
        print(f'[RESULT {cycle}] new_positive={n_new_positive} | total_unique_positive={self.history["n_positive"][-1]} | '
              f'cumulative_new_positive={self.history["n_positive_new_cumulative"][-1]} | '
              f'exploit={details.get("n_exploit", 0)} | explore={details.get("n_explore", 0)}', flush=True)

        self.model = self._build_model()
        return True

    def _save_results(self):
        if self.output_dir is None: return
        os.makedirs(self.output_dir, exist_ok=True)
        run_id = f'{self.input_name}_{self.task}_{self.model_type}_{self.acquisition_method}_seed{self.seed}_start{self.n_start}_batch{self.batch_size}'
        target_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(target_dir, exist_ok=True)
        acquired = pd.DataFrame(self.acquired_records)
        if acquired.duplicated(['Gene A', 'Gene B']).any(): raise AssertionError('实验包含跨轮次重复基因对')
        acquired.to_csv(os.path.join(target_dir, 'acquired_gene_pairs.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame(self.history).to_csv(os.path.join(target_dir, 'active_learning_history.csv'), index=False, encoding='utf-8-sig')
        init = dict(self.initialization_info)
        init.update({'input_file': os.path.basename(self.file_path), 'cell_line': self.cell_line,
                     'label_type': self.label_type, 'threshold_pct': self.threshold_pct,
                     'feature': self.feature_name, 'task': self.task, 'model': self.model_type,
                     'method': self.acquisition_method, 'seed': self.seed, 'n_start': self.n_start,
                     'fully_random_scheme': False,
                     'unchanged_random_draw': init.get('replacement', 'unknown') == 'none' if 'replacement' in init else 'unknown'})
        pd.DataFrame([init]).to_csv(os.path.join(target_dir, 'initialization_summary.csv'), index=False, encoding='utf-8-sig')
        print(f'[SAVED] {target_dir}', flush=True)

    def run(self, max_cycles: int = 5):
        self.prepare_data()
        train_idx, _ = self.handler.get_idx()
        self._record_history(0, n_new_positive=0)
        for cycle in range(1, max_cycles + 1):
            if not self.run_cycle(cycle): break
        self._save_results()
        return pd.DataFrame(self.history)
