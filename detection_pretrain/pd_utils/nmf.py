"""
NMF (Non-negative Matrix Factorization) 实现
从 Part-dependent Label Noise 项目复用
"""
import numpy as np


def norm(T):
    """矩阵行归一化"""
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def train_m(V, r, k, e):
    """
    非负矩阵分解: V ≈ W × H
    
    参数:
        V: (样本数 × 特征维度) 输入矩阵
        r: basis数量
        k: 最大迭代次数
        e: 误差阈值
    
    返回:
        W: (样本数 × r) 组合系数矩阵
        H: (r × 特征维度) 基矩阵
        error: 误差列表
    """
    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((r, n)))
    errors = []
    
    for iteration in range(k):
        V_pred = np.dot(W, H)
        E = V - V_pred
        err = np.sum(np.square(E))
        errors.append(err)
        
        if err < e:
            break
        
        # 更新 H
        a = np.dot(W.T, V)
        b = np.dot(np.dot(W.T, W), H)
        for i in range(r):
            for j in range(n):
                if b[i, j] != 0:
                    H[i, j] = H[i, j] * a[i, j] / b[i, j]
        
        # 更新 W
        c = np.dot(V, H.T)
        d = np.dot(np.dot(W, H), H.T)
        for i in range(m):
            for j in range(r):
                if d[i, j] != 0:
                    W[i, j] = W[i, j] * c[i, j] / d[i, j]
        
        print(f"Iteration {iteration + 1}, error: {err}")
        # 归一化 W
        W = norm(W)
    #print(W)
    return W, H, errors


def matrix_combination(basis_matrices, W, sample_idx, num_classes, basis):
    """
    构建单个样本的实例转移矩阵
    
    公式: M_i = Σ_{k=1}^{basis} W_{i,k} × A_k
    """
    W_sample = W[sample_idx, :]
    M = np.zeros((num_classes, num_classes))
    
    for i in range(basis):
        weight = float(W_sample[0, i])  # W是matrix类型
        M += weight * basis_matrices[i, :, :]
    
    # 清理接近0的值
    M[M < 1e-6] = 0.0
    
    return M