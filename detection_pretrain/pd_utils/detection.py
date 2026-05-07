"""
后门样本检测逻辑
"""

import numpy as np
import torch
import torch.nn.functional as F
from .nmf import matrix_combination


def extract_features(model, dataloader, feature_dim, batch_size):
    """
    提取模型特征表示（倒数第二层）
    
    返回:
        features: (N, feature_dim) numpy array
    """
    model.eval()
    features = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # 兼容不同模型接口
            outputs = model(data, revision=False)
            if isinstance(outputs, tuple):
                feature, _ = outputs  # (features, logits)
            else:
                # 如果模型没有返回features，尝试从倒数第二层提取
                feature = model.features(data) if hasattr(model, 'features') else outputs
            
            features.append(feature.cpu().numpy())
    
    features = np.vstack(features)
    return features


def detect_backdoor_samples(W, basis_matrices, suspected_target, num_classes, 
                           basis, threshold_percentile=95):
    """
    使用实例转移矩阵检测后门样本
    
    参数:
        W: (N, basis) NMF组合系数
        basis_matrices: (basis, num_classes, num_classes)
        suspected_target: 可疑的目标类
        num_classes: 类别数
        basis: 基的数量
        threshold_percentile: 异常阈值百分位数
    
    返回:
        suspicious_samples: 可疑样本信息列表
        all_scores: 所有样本的异常得分
    """
    num_samples = W.shape[0]
    anomaly_data = []
    
    for sample_idx in range(num_samples):
        # 构建实例转移矩阵
        instance_matrix = matrix_combination(
            basis_matrices, W, sample_idx, num_classes, basis
        )
        
        # 归一化
        row_sums = instance_matrix.sum(axis=1, keepdims=True)
        instance_matrix = instance_matrix / (row_sums + 1e-8)
        
        # === 特征1: 转移到目标类的最大概率 ===
        max_transfer_to_target = np.max(instance_matrix[:, suspected_target])
        
        # === 特征2: 转移矩阵的熵（后门样本熵低）===
        row_entropies = []
        for i in range(num_classes):
            row = instance_matrix[i, :]
            row = row + 1e-8  # 避免log(0)
            entropy = -np.sum(row * np.log(row))
            row_entropies.append(entropy)
        
        avg_entropy = np.mean(row_entropies)
        normalized_entropy = avg_entropy / np.log(num_classes)  # 归一化到[0,1]
        
        # === 特征3: NMF权重的集中度 ===
        W_sample = W[sample_idx, :].A.flatten()  # 转换matrix到array
        weight_max = np.max(W_sample)
        weight_mean = np.mean(W_sample)
        weight_concentration = weight_max - weight_mean
        
        # === 特征4: 转移矩阵中目标类列的集中度 ===
        target_column = instance_matrix[:, suspected_target]
        target_column_concentration = np.max(target_column)
        
        # === 特征5: 对角线占优程度 ===
        diagonal = np.diag(instance_matrix)
        diagonal_dominance = np.mean(diagonal)
        
        # === 综合评分 ===
        anomaly_score = (
            0.35 * max_transfer_to_target +           # 转移到目标类
            0.25 * (1 - normalized_entropy) +         # 低熵
            0.20 * weight_concentration +             # NMF权重异常
            0.15 * target_column_concentration +      # 目标类列集中
            0.05 * (1 - diagonal_dominance)           # 非对角占优
        )
        
        anomaly_data.append({
            'idx': sample_idx,
            'score': anomaly_score,
            'max_transfer': max_transfer_to_target,
            'entropy': normalized_entropy,
            'weight_conc': weight_concentration,
            'target_col_conc': target_column_concentration,
            'diag_dom': diagonal_dominance
        })
    
    # 找出异常样本
    all_scores = np.array([d['score'] for d in anomaly_data])
    threshold = np.percentile(all_scores, threshold_percentile)
    
    suspicious_samples = [
        d for d in anomaly_data if d['score'] > threshold
    ]
    
    # 按得分排序
    suspicious_samples.sort(key=lambda x: x['score'], reverse=True)
    
    return suspicious_samples, all_scores


def evaluate_detection(suspicious_indices, true_poison_indices, total_samples):
    """
    评估检测效果
    
    返回:
        metrics: 包含 precision, recall, f1 的字典
    """
    detected_set = set(suspicious_indices)
    poison_set = set(true_poison_indices)
    
    TP = len(detected_set & poison_set)
    FP = len(detected_set - poison_set)
    FN = len(poison_set - detected_set)
    TN = total_samples - TP - FP - FN
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }