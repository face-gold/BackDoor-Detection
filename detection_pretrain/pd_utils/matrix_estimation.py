"""
转移矩阵估计（后门检测版本）
改进自 Part-dependent Label Noise 的 fit() 函数
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def fit(X, num_classes, percentage, filter_outlier=False):
    """
    传入概率矩阵X, 为每个类别选出“最可信/最干净”的样本索引（即概率最大的前百分之几）
    当 filter_outlier=False(默认)，直接选取每一列（每个类别）概率最大的样本作为“锚点”样本。
    当 filter_outlier=True, 会先计算该类别概率的百分位阈值(eta_thresh), 把高于这个阈值的概率置为0,
    再在剩下的样本中选最大值，目的是排除极端异常值，选出更“稳健”的代表样本。

    """
    # number of classes
    c = num_classes
    T = np.empty((c, c)) # +1 -> index 
    eta_corr = X
    ind = []
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], percentage,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
            ind.append(idx_best)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
            
    return T, ind


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-1)
            
    return net


def estimate_transition_matrix_group(model, dataloader, num_classes, 
                                     basis=10, percentile_range=(95, 99)):
    """
    为后门检测估计转移矩阵组
    
    核心改进:
    1. 使用观测标签筛选样本（假设大部分标签正确）
    2. 使用百分位数避开异常高置信度样本
    3. 生成多个basis对应的转移矩阵
    
    参数:
        model: 后门模型
        dataloader: 数据加载器
        num_classes: 类别数
        basis: 基的数量
        percentile_range: 百分位数范围，如 (95, 99)
    
    返回:
        transition_matrix_group: (basis, num_classes, num_classes)
        idx_matrix_group: (num_classes, basis) 锚点样本索引
    """
    
    # Step 1: 提取所有样本的预测和标签
    print("  [1/3] Extracting predictions...")
    all_predictions = []
    all_labels = []
    
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            
            # 兼容不同模型接口
            if hasattr(model, 'forward'):
                outputs = model(data)
                if isinstance(outputs, tuple):
                    _, logits = outputs  # (features, logits)
                else:
                    logits = outputs
            else:
                logits = model(data)
            
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_predictions)  # (N, num_classes)
    labels = np.concatenate(all_labels)       # (N,)
    print(f"      Extracted {len(labels)} labels")
    
    print(f"      Extracted {len(predictions)} samples")
    
    # Step 2: 为每个basis构建转移矩阵
    print("  [2/3] Estimating transition matrices...")
    
    transition_matrix_group = np.zeros((basis, num_classes, num_classes))
    idx_matrix_group = np.zeros((num_classes, basis), dtype=int)
    
    # 使用不同的百分位数
    percentiles = np.linspace(percentile_range[0], percentile_range[1], basis)
    
    for basis_idx, percentile in enumerate(percentiles):
        T = np.zeros((num_classes, num_classes))
        
        for class_i in range(num_classes):
            # 关键改进：只从标签为 class_i 的样本中选择
            mask = (labels == class_i)
            class_samples_pred = predictions[mask]
            class_samples_idx = np.where(mask)[0]
            
            if len(class_samples_pred) == 0:
                # 如果该类没有样本，使用均匀分布
                print(f"      Warning: No samples for class {class_i}, using uniform distribution.")
                T[class_i, :] = 1.0 / num_classes
                idx_matrix_group[class_i, basis_idx] = -1
                continue
            
            # 计算样本在类别 class_i 上的置信度
            confidences = class_samples_pred[:, class_i]
            
            # 使用百分位数过滤（避开过于自信的样本）
            threshold = np.percentile(confidences, percentile)
            
            # 选择置信度接近阈值的样本
            # 方法1: 严格百分位数
            # selected_mask = confidences <= threshold
            
            # 方法2: 范围选择（更鲁棒）
            margin = 0.05
            selected_mask = (confidences >= threshold - margin) & (confidences <= threshold + margin)
            
            if selected_mask.sum() == 0:
                # 如果没有符合条件的样本，选择置信度最接近阈值的
                closest_idx = np.argmin(np.abs(confidences - threshold))
                selected_samples = class_samples_pred[closest_idx:closest_idx+1]
                anchor_idx = class_samples_idx[closest_idx]
            else:
                selected_samples = class_samples_pred[selected_mask]
                selected_indices = class_samples_idx[selected_mask]
                
                # 从符合条件的样本中选择置信度最高的作为锚点
                best_in_selected = np.argmax(selected_samples[:, class_i])
                anchor_idx = selected_indices[best_in_selected]
            
            # 平均选中样本的预测分布
            T[class_i, :] = selected_samples.mean(axis=0)
            idx_matrix_group[class_i, basis_idx] = anchor_idx
        
        # 归一化
        T = T / (T.sum(axis=1, keepdims=True) + 1e-8)
        transition_matrix_group[basis_idx] = T
    
    print(f"      Generated {basis} transition matrices")
    
    return transition_matrix_group, idx_matrix_group


def analyze_transition_matrices(transition_matrix_group, num_classes):
    """
    分析转移矩阵，检测可疑的目标类
    
    返回:
        suspected_target: 最可疑的目标类
        anomaly_scores: 每个类的异常得分
    """
    anomaly_scores = []
    
    for target_class in range(num_classes):
        # 统计转移到该类的异常模式
        suspicious_sources = 0
        total_incoming_transfer = 0
        max_incoming = 0
        
        for basis_idx in range(len(transition_matrix_group)):
            T = transition_matrix_group[basis_idx]
            
            # 检查非对角线元素
            for source_class in range(num_classes):
                if source_class == target_class:
                    continue
                
                transfer_prob = T[source_class, target_class]
                total_incoming_transfer += transfer_prob
                
                if transfer_prob > 0.15:  # 阈值：超过15%认为异常
                    suspicious_sources += 1
                
                max_incoming = max(max_incoming, transfer_prob)
        
        # 计算对角线元素的平均值
        diagonal_avg = np.mean([T[target_class, target_class] 
                               for T in transition_matrix_group])
        
        anomaly_scores.append({
            'class': target_class,
            'suspicious_sources': suspicious_sources,
            'total_transfer': total_incoming_transfer,
            'max_incoming': max_incoming,
            'avg_diagonal': diagonal_avg,
            'anomaly_score': total_incoming_transfer * suspicious_sources
        })
    
    # 按异常得分排序
    anomaly_scores.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    return anomaly_scores[0]['class'], anomaly_scores