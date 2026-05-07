"""
基矩阵学习
从 Part-dependent Label Noise 项目复用
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


def norm_tensor(T):
    """Tensor版本的归一化"""
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm


class MatrixOptimize(nn.Module):
    """
    基矩阵优化模型
    用于学习 basis 个可学习的转移矩阵
    """
    def __init__(self, basis_num, num_classes):
        super(MatrixOptimize, self).__init__()
        self.basis_matrix = self._make_layer(basis_num, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-1)
    
    def _make_layer(self, basis_num, num_classes):
        layers = []
        for i in range(basis_num):
            layers.append(nn.Linear(num_classes, 1, bias=False))
        return nn.Sequential(*layers)
    
    def forward(self, W, num_classes):
        """
        前向传播：用基矩阵的线性组合重构转移矩阵
        
        参数:
            W: (basis,) 组合系数
            num_classes: 类别数
        
        返回:
            result: (num_classes, 1) 重构的转移矩阵的一行
        """
        results = torch.zeros(num_classes, 1)
        
        for i in range(len(W)):
            # 构建对角系数矩阵
            coeff_matrix = float(W[i]) * torch.eye(num_classes, num_classes)
            
            # 归一化权重（满足约束）
            self.basis_matrix[i].weight.data = norm_tensor(
                self.basis_matrix[i].weight.data
            )
            
            # 计算贡献
            anchor_vector = self.basis_matrix[i](coeff_matrix)
            results += anchor_vector
            
            # 再次归一化
            self.basis_matrix[i].weight.data = norm_tensor(
                self.basis_matrix[i].weight.data
            )
        
        return results


def learn_basis_matrices(W, transition_matrix_group, idx_matrix_group, 
                        num_classes, basis, epochs=1500, lr=0.001,
                        early_stop_threshold=0.02):
    """
    学习基转移矩阵
    
    参数:
        W: (N, basis) 样本的组合系数
        transition_matrix_group: (basis, num_classes, num_classes)
        idx_matrix_group: (num_classes, basis) 锚点索引
        num_classes: 类别数
        basis: 基的数量
        epochs: 训练轮数
        lr: 学习率
        early_stop_threshold: 早停阈值
    
    返回:
        basis_matrices: (basis, num_classes, num_classes)
    """
    basis_matrices = np.zeros((basis, num_classes, num_classes))
    criterion = nn.MSELoss()
    
    print(f"  [3/3] Learning basis matrices...")
    
    for class_i in range(num_classes):
        # 为每个类别单独训练
        model = MatrixOptimize(basis, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 重新初始化
        for m in model.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-1)
        
        for epoch in range(epochs):
            loss_total = 0.0
            
            for basis_idx in range(basis):
                # 获取锚点样本的组合系数
                anchor_idx = int(idx_matrix_group[class_i, basis_idx])
                
                if anchor_idx == -1:  # 该类没有样本
                    continue
                
                W_anchor = W[anchor_idx, :]  # numpy matrix
                W_anchor_list = [float(W_anchor[0, j]) for j in range(basis)]
                
                # 目标：该类在该basis下的转移概率
                target = transition_matrix_group[basis_idx, class_i, :]
                target = torch.from_numpy(target[:, np.newaxis]).float()
                
                # 预测
                prediction = model(W_anchor_list, num_classes)
                
                # 计算损失
                loss = criterion(prediction, target)
                loss_total += loss
            
            if loss_total > 0:
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
            
            # 早停
            if loss_total.item() < early_stop_threshold:
                break
        
        # 提取学到的基矩阵
        for basis_idx in range(basis):
            basis_matrices[basis_idx, class_i, :] = (
                model.basis_matrix[basis_idx].weight.data.numpy().flatten()
            )
    
    # 清理接近0的值
    basis_matrices[basis_matrices < 1e-6] = 0.0
    
    print(f"      Learned {basis} basis matrices for {num_classes} classes")
    
    return basis_matrices