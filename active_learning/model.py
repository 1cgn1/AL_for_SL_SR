import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from deepforest import CascadeForestClassifier
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型类
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
        logits = torch.log(probs)
        return logits

class LREnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42, max_iter=3000,n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [
            LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
                random_state=s,
                n_jobs=n_jobs
            )
            for s in self.seeds
        ]
    def __repr__(self):
        return f"LREnsemble(size={self.ensemble_size})"

class RFEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, n_estimators=100, max_depth=None, seed=42,n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                random_state=s,
                n_jobs= n_jobs
            )
            for s in self.seeds
        ]
    def __repr__(self):
        return f"RFEnsemble(size={self.ensemble_size})"

class SVMEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42, C=0.1, gamma="scale"):
        super().__init__(ensemble_size, seed)

        self.models = [
            CalibratedClassifierCV(
                estimator=LinearSVC(random_state=s,
                                    C=C,
                                    tol=1e-3,
                                    dual=True,   # 只有在此参数为True的时候random_state才生效
                                    max_iter=5000)
                , method='sigmoid',
                cv=3
            )
            for s in self.seeds
        ]

    def __repr__(self):
        return f"SVMEnsemble(size={self.ensemble_size})"

class XGBEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42,n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [
            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=8,
                reg_lambda=1,
                random_state=s,
                n_jobs= n_jobs,
                device="cpu"
            )
            for s in self.seeds
        ]

    def __repr__(self):
        return f"XGBEnsemble(size={self.ensemble_size})"

class DFEnsemble(BaseEnsemble):
    def __init__(self, ensemble_size=10, seed=42,n_jobs=10):
        super().__init__(ensemble_size, seed)
        self.models = [
            CascadeForestClassifier(
                random_state=s,
                n_jobs= n_jobs
            )for s in self.seeds]


    def __repr__(self):
        return f"DFEnsemble(size={self.ensemble_size})"

#dnn的方法和其他的机器学习不太一样，不能直接继承之前的base ensemble，需要重写,这里是定义单个模型
class DNNClassifier(nn.Module):
    def __init__(self,input_dim=1000,drop_out_p=0.1): #input_dim依赖外部传入，此处的数值无意义
        super().__init__()
        hidden_dims = [2048,1024,512,256,128,64,32,16,8,4]
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

#ensemble_size:ensemble中的模型数量，seed:随机种子，input_dim:输入特征的维度，Learning_rate:模型超参数，学习率
# drop_out:模型超参数，device:模型运行的设备，epochs:每次输入模型会训练多少轮
class DNNEnsemble():
    def __init__(self, ensemble_size=10, seed=42,input_dim=1000,learning_rate=0.1,drop_out_p=0.1
                 ,device=device,epochs=5):
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(0, 10000, ensemble_size)
        self.models = []
        self.optimizers = []
        self.epochs = epochs
        self.device = device
        self.input_dim = input_dim

        for i in range(self.ensemble_size):
            seed = self.seeds[i]
            torch.manual_seed(seed)
            model = DNNClassifier(input_dim=self.input_dim, drop_out_p=drop_out_p)
            model = model.to(self.device)
            self.models.append(model)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            self.optimizers.append(optimizer)

    def train(self, X, y):
        for ensemble_index in range(self.ensemble_size):
            for epoch in range(self.epochs):
                model = self.models[ensemble_index]
                optimizer = self.optimizers[ensemble_index]
                if isinstance(X,torch.Tensor):
                    X = X.to(self.device).float()
                else:
                    X = torch.tensor(X).to(self.device).float()
                    y = torch.tensor(y).to(self.device)
                optimizer.zero_grad()
                output = model(X)
                loss = nn.functional.cross_entropy(output,y)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X).to(self.device).float()
            eps = 1e-12
            probs = []
            for model in self.models:
                model.eval()
                output = model(X).to(self.device)
                probs.append(nn.functional.softmax(output,dim=1))
            probs = torch.stack(probs,dim=1)
            logits = torch.log(probs+eps)
            logits = logits.to('cpu')
        return logits

def build_model(model_type, seed,n_jobs,input_dim,learning_rate,drop_out_p,
                epochs,params):
    if model_type == "rf":
        return RFEnsemble(seed=seed,n_jobs=n_jobs, **params)
    elif model_type == "lr":
        return LREnsemble(seed=seed,n_jobs=n_jobs, **params)
    elif model_type == "svm":
        return SVMEnsemble(seed=seed, **params)
    elif model_type == "xgb":
        return XGBEnsemble(seed=seed,n_jobs=n_jobs, **params)
    elif model_type == "df":
        return DFEnsemble(seed=seed,n_jobs=n_jobs, **params)
    elif model_type == "dnn":
        return DNNEnsemble(seed=seed, input_dim=input_dim,learning_rate=learning_rate,
        drop_out_p=drop_out_p,epochs=epochs,**params)
    else:
        raise ValueError("Unknown model type")

