
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_counts(y):
    labels = np.asarray(y).reshape(-1)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos + n_neg != len(labels): raise ValueError('训练标签必须为二分类0/1')
    if n_pos == 0 or n_neg == 0: raise ValueError(f'模型训练需要正负两类：positive={n_pos}, negative={n_neg}')
    return labels, n_pos, n_neg

class BaseEnsemble:
    def __init__(self, ensemble_size=10, seed=42):
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(0, 10000, ensemble_size)

    def train(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):

        eps = 1e-10
        probs = []
        for m in self.models:
            p = m.predict_proba(X) + eps
            probs.append(torch.tensor(p, dtype=torch.float32))
        probs = torch.stack(probs, dim=1)
        return torch.log(probs)

class LREnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42, max_iter=3000, n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [LogisticRegression(max_iter=max_iter, solver='lbfgs', class_weight='balanced', random_state=int(s), n_jobs=n_jobs) for s in self.seeds]

    def train(self, X, y):

        labels, n_pos, n_neg = get_class_counts(y)
        pos_idx, neg_idx = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        for k, model in enumerate(self.models):
            rng = np.random.default_rng(int(self.seeds[k]))

            boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
            rng.shuffle(boot_idx)
            X_boot = X.iloc[boot_idx] if hasattr(X, 'iloc') else X[boot_idx]
            model.fit(X_boot, labels[boot_idx])
        print(f'[LR Bootstrap] positive={n_pos}, negative={n_neg}, members={len(self.models)}', flush=True)

    def __repr__(self):
        return f'LREnsemble(size={self.ensemble_size})'

class RFEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, n_estimators=100, max_depth=None, seed=42, n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=int(s), n_jobs=n_jobs) for s in self.seeds]

    def train(self, X, y):
        labels, n_pos, n_neg = get_class_counts(y)
        ratio = n_neg / n_pos

        weights = {0: 1.0, 1: float(ratio)}
        for model in self.models:
            model.set_params(class_weight=weights)
            model.fit(X, labels)
        print(f'[RF weight] positive={n_pos}, negative={n_neg}, class_weight={weights}', flush=True)

    def __repr__(self):
        return f'RFEnsemble(size={self.ensemble_size})'

class XGBEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42, n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric='logloss', scale_pos_weight=1.0, reg_lambda=1, random_state=int(s), n_jobs=n_jobs) for s in self.seeds]

    def train(self, X, y):
        labels, n_pos, n_neg = get_class_counts(y)
        ratio = float(n_neg / n_pos)
        for model in self.models:
            model.set_params(scale_pos_weight=ratio)
            model.fit(X, labels)
        print(f'[XGB weight] positive={n_pos}, negative={n_neg}, scale_pos_weight={ratio:.6f}', flush=True)

    def __repr__(self):
        return f'XGBEnsemble(size={self.ensemble_size})'

class DFEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42, n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [CascadeForestClassifier(random_state=int(s), n_jobs=n_jobs) for s in self.seeds]

    def __repr__(self):
        return f'DFEnsemble(size={self.ensemble_size})'

class DNNClassifier(nn.Module):
    def __init__(self, input_dim=1000, drop_out_p=0.1):
        super().__init__()
        hidden_dims = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out_p))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DNNEnsemble:
    def __init__(self, ensemble_size=10, seed=42, input_dim=1000, learning_rate=0.1, drop_out_p=0.1, device=device, epochs=5):
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(0, 10000, ensemble_size)
        self.models = []
        self.optimizers = []
        self.epochs = epochs
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.drop_out_p = drop_out_p

    def train(self, X, y):

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device).reshape(-1)
        if X.ndim != 2 or X.shape[1] != self.input_dim or len(X) != len(y): raise ValueError('DNN输入特征维度或样本数不一致')

        self.models = []
        self.optimizers = []
        for s in self.seeds:
            torch.manual_seed(int(s))
            model = DNNClassifier(input_dim=self.input_dim, drop_out_p=self.drop_out_p).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            self.models.append(model)
            self.optimizers.append(optimizer)
        for model, optimizer in zip(self.models, self.optimizers):

            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
        print(f'[DNN fresh train] members={len(self.models)}, samples={len(y)}, epochs={self.epochs}', flush=True)

    def predict(self, X):
        if not self.models: raise RuntimeError('DNN尚未训练，请先调用train(X, y)')
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        probs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(X)
                probs.append(F.softmax(output, dim=1))
            probs = torch.stack(probs, dim=1)
            logits = torch.log(probs + 1e-12)
        return logits.cpu()

    def __repr__(self):
        return f'DNNEnsemble(size={self.ensemble_size})'

def build_model(model_type, seed, n_jobs, input_dim, learning_rate, drop_out_p, epochs, params):
    if model_type == 'rf':
        return RFEnsemble(seed=seed, n_jobs=n_jobs, **params)
    elif model_type == 'lr':
        return LREnsemble(seed=seed, n_jobs=n_jobs, **params)
    elif model_type == 'xgb':
        return XGBEnsemble(seed=seed, n_jobs=n_jobs, **params)
    elif model_type == 'df':
        return DFEnsemble(seed=seed, n_jobs=n_jobs, **params)
    elif model_type == 'dnn':
        return DNNEnsemble(seed=seed, input_dim=input_dim, learning_rate=learning_rate, drop_out_p=drop_out_p, epochs=epochs, **params)
    else:
        raise ValueError('Unknown model type')
