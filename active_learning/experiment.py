from active_learning.get_data import get_data
from active_learning.Handler import Handler
from active_learning.acquisition import Acquisition
import active_learning.model as model
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split




# 定义实验类
class Experiment:
    # file_path:数据文件路径，feature_cols:特征列，test_size:测试集比例，n_start:初始训练集样本数量，batch_size:每轮主动学习选择的样本数，max_screen_size：最多训练样本数，
    # acquisition_method：主动学习采样方法，rf_params：随机森林参数，seed：随机种子，n_jobs:模型使用的cpu核心数量
    #learning_rate:仅对dnn生效，学习率超参数，drop_out_p:仅对dnn生效，超参数，epochs:仅对dnn生效，每次学多少轮
    #feature_type:特征数据格式，通路特征传入int8，esm特征传入float32
    def __init__(self,file_path: str,feature_cols: list,test_size: float = 0.2,n_start: int = 64,batch_size: int = 32,
                 max_screen_size: int = None,acquisition_method: str = "random",model_type="rf"
                 , model_params=None,seed: int = 42,n_jobs=10,learning_rate: float = 0.01,drop_out_p: float = 0.2,
                 epochs: int = 5,positive_label: int=0,feature_type='int8',pool_ratio: float=1,save_genes: bool=False,**params):
        self.file_path = file_path
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.n_start = n_start
        self.batch_size = batch_size
        self.max_screen_size = max_screen_size
        self.acquisition_method = acquisition_method
        self.seed = seed
        self.model_type = model_type
        self.n_jobs = n_jobs
        self.input_dim = len(self.feature_cols)
        self.learning_rate = learning_rate
        self.drop_out_p = drop_out_p
        self.epochs = epochs
        self.positive_label = positive_label
        self.feature_type = feature_type
        self.pool_ratio = pool_ratio
        self.save_genes = save_genes
        # 如果 model_params 没有传入，使用空字典，保存到类属性
        if model_params is None:
            model_params = {}

        self.model_params = model_params
        # 占位属性
        if self.save_genes:
            self.genes = pd.DataFrame()
        self.df_screen = None
        self.df_pool = None
        self.df_test = None
        self.handler = None
        self.model = None
        self.acquisition = None
        # 日志字典，用于记录每轮 AL 的指标，包含：轮次、训练集大小、池大小、正样本数量、起始集规模
        self.history = {"cycle": [],"train_size": [],"pool_size": [],"n_positive": [],
                        "n_start": [], "pos_neg_ratio": []}
    # 准备数据
    def prepare_data(self):
        df_train, df_pool, df_test = get_data(file_path=self.file_path,test_size=self.test_size,n_start=self.n_start,
                                              random_state=self.seed, feature_cols=self.feature_cols
                                              ,positive_label=self.positive_label,feature_type=self.feature_type)
        self.df_train = df_train
        self.df_pool = df_pool
        self.df_test = df_test

        # 计算初始训练集正负样本比例
        n_pos = (df_train["y"] == 1).sum()
        n_neg = (df_train["y"] == 0).sum()

        self.pos_ratio = n_pos / (n_pos + n_neg)
        self.pos_neg_ratio = f"{n_pos}:{n_neg}"

        self.X_test = df_test[self.feature_cols].values
        self.y_test = df_test["y"].values

        # 全体标签，合并初始训练集和池，得到完整 screen 数据
        df_screen = pd.concat([df_train, df_pool], axis=0).reset_index(drop=True)
        self.df_screen = df_screen
        # 提取全体标签 y_all，用于 Handler 管理
        y_all = df_screen["y"].values

        train_idx = np.arange(len(df_train))
        pool_idx = np.arange(len(df_train), len(df_screen))
        self.handler = Handler(train_idx, pool_idx, y_all)

        # 选择获取函数
        self.acquisition = Acquisition(method=self.acquisition_method,seed=self.seed)
        self.acquisition.df_screen = self.df_screen
        self.acquisition.feature_cols = self.feature_cols
        self.acquisition.handler = self.handler

        # 选择模型
        self.model = model.build_model(model_type=self.model_type,seed=self.seed,params=self.model_params
                                       ,n_jobs=self.n_jobs,input_dim=self.input_dim,learning_rate=self.learning_rate,
                                       drop_out_p=self.drop_out_p,epochs=self.epochs)

        # 如果 max_screen_size 没传入，设置为整个 screen 数据大小，防止训练集超过总样本数
        if self.max_screen_size is None:
            self.max_screen_size = len(df_screen)

    # 定义执行单轮主动学习的方法
    def run_cycle(self, cycle: int):
        # 获取当前训练集和未标记样本池的索引
        train_idx, pool_idx = self.handler.get_idx()
        # 如果池为空，停止循环；如果训练集达到最大数量，停止循环
        if len(pool_idx) == 0:
            print("Pool empty, stopping")
            return False
        if len(train_idx) >= self.max_screen_size:
            print("Reached max_screen_size, stopping")
            return False
        # 每轮采样数量不能超过池的大小
        current_batch = min(self.batch_size, len(pool_idx))
        # 如果采样数量≤0，则停止
        if current_batch <= 0:
            print("current_batch <= 0, stopping")
            return False
        # 准备数据
        X_train = self.df_screen.iloc[train_idx][self.feature_cols].values
        y_train = self.df_screen.iloc[train_idx]["y"].values
        X_pool = self.df_screen.iloc[pool_idx][self.feature_cols].values
        #在pool里抽样，提高运行速度
        if self.pool_ratio == 1:
            X_pool_one_cycle = X_pool
            pool_idx_one_cycle = pool_idx
        else:
            #防止train_test_split输入的标签只有一种导致报错
            try:
                X_pool_one_cycle,useless = train_test_split(self.df_screen.iloc[pool_idx]
                                                            ,stratify=self.df_screen.iloc[pool_idx,:]['y']
                                                            ,shuffle=True,random_state=self.seed + cycle
                                                            ,train_size=self.pool_ratio)
            except ValueError:
                print('缺少正样本')
                X_pool_one_cycle, useless = train_test_split(self.df_screen.iloc[pool_idx]
                                                             , shuffle=True, random_state=self.seed + cycle
                                                             , train_size=self.pool_ratio)
            pool_idx_one_cycle = X_pool_one_cycle.index.copy()
            X_pool_one_cycle = X_pool_one_cycle[self.feature_cols].values
        # 训练当前模型
        print(len(X_pool_one_cycle))
        self.model.train(X_train, y_train)
        # 对池中的样本预测 logits，用于采样选择
        logits = self.model.predict(X_pool_one_cycle)
        # 根据获取函数从池中选择样本
        picks = self.acquisition(logits_N_K_C=logits,pool_idx=pool_idx_one_cycle,n=current_batch)
        # 如果没有选择任何样本，停止循环
        if len(picks) == 0:
            print("No picks returned, stopping")
            return False
        # 更新 handler 内部索引
        self.handler.add(picks)
        #保存每一轮选中的样本名称
        if self.save_genes:
            train_index,useless = self.handler.get_idx()
            genes_data = self.df_screen.iloc[train_index].iloc[:,:3].copy()
            genes_data[['cycle','n_start', 'batch_size', 'positive_label', 'model', 'acquisition_function']]= (cycle,
                                                                                                             self.n_start,
                                                                                                      self.batch_size,
                                                                                                      self.positive_label,
                                                                                                      self.model_type,
                                                                                                      self.acquisition_method)
            self.genes = pd.concat([self.genes,genes_data],axis=0,ignore_index=True)
        # 记录每轮指标到 history 字典
        self.history["n_start"].append(self.n_start)
        self.history["cycle"].append(cycle)
        self.history["train_size"].append(len(self.handler.train_idx))
        self.history["pool_size"].append(len(self.handler.pool_idx))
        self.history["n_positive"].append(self.df_screen.iloc[self.handler.train_idx]["y"].sum())
        self.history["pos_neg_ratio"].append(self.pos_neg_ratio)

        # 打印本轮循环的指标，方便观察训练进度
        print(
            f"Cycle {cycle:02d} | "
            f"Train={len(self.handler.train_idx):4d} | "
            f"Pool={len(self.handler.pool_idx):4d} | "
            f"Positives={self.history['n_positive'][-1]:4d} | "
)
        #重新定义模型，确保在主动学习的每轮循环里模型都是cold start的
        self.model = model.build_model(model_type=self.model_type,seed=self.seed,params=self.model_params
                                       ,n_jobs=self.n_jobs,input_dim=self.input_dim,learning_rate=self.learning_rate,
                                       drop_out_p=self.drop_out_p,epochs=self.epochs)

        return True
    # 定义运行完整主动学习实验的方法，循环执行 max_cycles 轮，如果某轮返回 False ，停止循环
    def run(self, max_cycles: int = 50):
        self.prepare_data()

        # 先记录初始状态（cycle = 0）
        train_idx, pool_idx = self.handler.get_idx()
        if self.save_genes:
            genes_data = self.df_train.iloc[:,:3].copy()
            genes_data[['cycle','n_start', 'batch_size', 'positive_label', 'model', 'acquisition_function']]= [0,
                                                                                                             self.n_start,
                                                                                                      self.batch_size,
                                                                                                      self.positive_label,
                                                                                                      self.model_type,
                                                                                                      self.acquisition_method]
            self.genes = pd.concat([self.genes,genes_data],axis=0)
        self.history["cycle"].append(0)
        self.history["train_size"].append(len(train_idx))
        self.history["pool_size"].append(len(pool_idx))
        self.history["n_positive"].append(
            self.df_screen.iloc[train_idx]["y"].sum()
        )
        self.history["n_start"].append(self.n_start)
        self.history["pos_neg_ratio"].append(self.pos_neg_ratio)

        # 从第1轮开始
        for cycle in range(1, max_cycles + 1):
            cont = self.run_cycle(cycle)
            if not cont:
                break

        if self.save_genes:
            return pd.DataFrame(self.history),self.genes
        else:
            return pd.DataFrame(self.history)