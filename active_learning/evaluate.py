from sklearn.metrics import (precision_score, recall_score, f1_score, average_precision_score, confusion_matrix)
import torch
import numpy as np

def evaluate_binary_classification(logits: torch.Tensor, y_true: np.ndarray):
    probs = torch.exp(logits)

    # 对每个模型计算指标
    n_models = probs.shape[1]
    precision_list, recall_list, f1_list, aupr_list = [], [], [], []

    for k in range(n_models):
        # 获取第k个模型的概率
        model_probs = probs[:, k, :]
        # 正类概率
        y_score = model_probs[:, 1].cpu().numpy()
        # 使用阈值0.5进行二分类，预测概率大于0.5为正样本
        y_pred = (y_score >= 0.5).astype(int)
        # 计算指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        aupr = average_precision_score(y_true, y_score)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        aupr_list.append(aupr)

    # 返回平均指标
    return {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list),
        "aupr": np.mean(aupr_list),
        "precision_std": np.std(precision_list),
        "recall_std": np.std(recall_list),
        "f1_std": np.std(f1_list)
    }