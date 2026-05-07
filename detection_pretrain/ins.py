"""
Instance Matrix Backdoor Detection
基于实例转移矩阵的后门样本检测方法
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import pandas as pd
from pprint import pformat
import yaml
import pickle
from tqdm import tqdm
import logging
import copy
from torch import nn

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result
from utils.trainer_cls import PureCleanModelTrainer
from utils.log_assist import get_git_info
from utils.bd_dataset_v2 import xy_iter
from torch.utils.data import DataLoader
from defense.base import defense

# 导入检测相关工具
from detection_pretrain.pd_utils import (
    train_m,
    fit,
    init_params
)
from detection_pretrain.pd_utils.basis_learning import MatrixOptimize


class InstanceMatrix():
    """Instance Matrix 检测核心算法"""
    name: str = "Instance Matrix Backdoor Detection"
    
    def __init__(self, num_classes, feature_dim=512, basis=10, 
                 nmf_iter=10, nmf_threshold=1e-5, 
                 matrix_epochs=10, detection_threshold=95):
        """
        参数:
            num_classes: 类别数
            feature_dim: 特征维度  
            basis: 基矩阵数量
            nmf_iter: NMF迭代次数
            nmf_threshold: NMF收敛阈值
            matrix_epochs: 基矩阵学习轮数
            detection_threshold: 检测阈值百分位数
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.basis = basis
        self.nmf_iter = nmf_iter
        self.nmf_threshold = nmf_threshold
        self.matrix_epochs = matrix_epochs
        self.detection_threshold = detection_threshold
    
    def detect_backdoor_samples(self, W, basis_matrices, num_classes, 
                               basis, threshold_percentile=95):
        """
        使用实例转移矩阵检测后门样本
        
        参数:
            W: (N, basis) NMF组合系数
            basis_matrices: (basis, num_classes, num_classes)
            num_classes: 类别数
            basis: 基的数量
            threshold_percentile: 异常阈值百分位数
        
        返回:
            suspicious_samples: 可疑样本信息列表
            all_scores: 所有样本的异常得分
            all_matrices: 所有样本的实例转移矩阵
        """
        num_samples = W.shape[0]
        anomaly_data = []
        all_matrices = {}  # 保存所有样本的实例转移矩阵
        
        print(f"  Processing {num_samples} samples...")
        
        for sample_idx in tqdm(range(num_samples), desc="Computing instance matrices"):
            # 构建实例转移矩阵
            instance_matrix = self.matrix_combination(
                basis_matrices, W, sample_idx, num_classes, basis
            )
            
            # 归一化
            row_sums = instance_matrix.sum(axis=1, keepdims=True)
            instance_matrix_norm = instance_matrix / (row_sums + 1e-8)
            
            # 保存实例转移矩阵
            all_matrices[sample_idx] = {
                'raw_matrix': instance_matrix.copy(),
                'normalized_matrix': instance_matrix_norm.copy()
            }
            
            # === 特征1: 对角线占优程度（正常样本对角线应该占优）===
            # 计算方法：取归一化转移矩阵的对角线元素（即每一类转移到自身的概率），然后求平均
            # 正常样本的转移矩阵应以对角为主（即大概率留在本类），后门样本可能对角占优降低。
            diagonal = np.diag(instance_matrix_norm)
            diagonal_dominance = np.mean(diagonal)
            
            # === 特征2: 转移矩阵的熵（后门样本熵可能异常）===
            # 计算方法：对每一行（每个类别的转移概率分布）计算信息熵，然后对所有行取平均，最后归一化到 [0,1]
            # 熵越高，分布越均匀；正常样本熵适中，后门样本可能熵异常（过高或过低）
            row_entropies = []
            for i in range(num_classes):
                row = instance_matrix_norm[i, :]
                row = row + 1e-8  # 避免log(0)
                entropy = -np.sum(row * np.log(row))
                row_entropies.append(entropy)
            
            avg_entropy = np.mean(row_entropies)
            normalized_entropy = avg_entropy / np.log(num_classes)  # 归一化到[0,1]
            
            # === 特征3: 非对角元素的最大值（后门样本可能有异常转移）===
            # 计算方法：将对角线元素置零，取剩下所有元素的最大值。
            # 后门样本可能存在异常大的非对角转移概率（即某一类异常地转移到另一类）
            non_diag_matrix = instance_matrix_norm - np.diag(diagonal)
            max_off_diagonal = np.max(non_diag_matrix)
            
            # === 特征4: NMF权重的集中度 ===
            # 计算方法：取该样本的NMF权重向量，计算最大值与均值的差值
            # 如果权重高度集中（即主要由一个基矩阵主导），可能是异常样本
            W_sample = W[sample_idx, :].A.flatten()  # 转换matrix到array
            weight_max = np.max(W_sample)
            weight_mean = np.mean(W_sample)
            weight_concentration = weight_max - weight_mean
            
            # === 特征5: 矩阵的稀疏性（非零元素比例）===
            # 归一化矩阵中大于 1e-6 的元素占总元素数的比例
            # 反映转移矩阵的稀疏程度，异常样本可能更稠密或更稀疏
            non_zero_ratio = np.sum(instance_matrix_norm > 1e-6) / (num_classes * num_classes)
            
            # === 特征6: 转移矩阵的标准差（衡量分布的分散程度）===
            # 归一化转移矩阵所有元素的标准差
            # 衡量分布的分散程度，异常样本的分布可能更极端
            matrix_std = np.std(instance_matrix_norm)
            
            # === 综合评分（不依赖目标类）===
            # 异常样本特征：对角线占优低、熵异常、有大的非对角转移、权重集中等
            anomaly_score = (
                0.30 * (1 - diagonal_dominance) +        # 对角占优低 -> 异常
                0.25 * max_off_diagonal +                # 大的非对角转移 -> 异常  
                0.20 * weight_concentration +            # NMF权重集中 -> 异常
                0.15 * abs(normalized_entropy - 0.5) +   # 熵偏离中值 -> 异常
                0.10 * matrix_std                        # 高标准差 -> 异常
            )
            
            anomaly_data.append({
                'idx': sample_idx,
                'score': anomaly_score,
                'diagonal_dominance': diagonal_dominance,
                'entropy': normalized_entropy,
                'max_off_diagonal': max_off_diagonal,
                'weight_concentration': weight_concentration,
                'non_zero_ratio': non_zero_ratio,
                'matrix_std': matrix_std
            })
        
        # 找出异常样本
        all_scores = np.array([d['score'] for d in anomaly_data])
        threshold = np.percentile(all_scores, threshold_percentile)
        
        suspicious_samples = [
            d for d in anomaly_data if d['score'] > threshold
        ]
        
        # 按得分排序
        suspicious_samples.sort(key=lambda x: x['score'], reverse=True)
        
        return suspicious_samples, all_scores, all_matrices, anomaly_data
    
    def detect(self, model, dataloader, poison_indices=None, save_dir=None):
        """
        执行检测
        
        参数:
            model: 后门模型
            dataloader: 数据加载器
            poison_indices: 真实的后门样本索引(用于评估)
            save_dir: 结果保存路径
        
        返回:
            suspicious_samples: 检测到的可疑样本
            metrics: 评估指标(如果提供了poison_indices)
            all_matrices: 所有样本的实例转移矩阵
        """
        print("  [a] Extracting features...")
        features = self.get_features(model, dataloader)
        print(f"    Feature shape: {features.shape}")
        print("前5个样本的特征：")
        print(features[:5])
        
        print("  [b] Performing NMF decomposition...")
        W, H, errors = train_m(
            features, 
            self.basis, 
            self.nmf_iter, 
            self.nmf_threshold
        )
        print(f"    W shape: {W.shape}, H shape: {H.shape}")
        print(f"    Final reconstruction error: {errors[-1]:.6f}")
        
        print("  [c] Estimating part-transition matrices...")
        logits_matrix = self.get_probability(model, dataloader)
        # 输出前5个样本的logits
        print("前5个样本的logits：")
        print(logits_matrix[:5])
        idx_matrix_group, transition_matrix_group = self.estimate_matrix(logits_matrix, model_save_dir=save_dir, exclude_indices=poison_indices)
        func=nn.MSELoss() # 损失函数

        matrix_model = MatrixOptimize(self.basis, self.num_classes)
        optimizer_1 = torch.optim.Adam(matrix_model.parameters(), lr=0.001)
        basis_matrices = self.basis_matrix_optimize(
            matrix_model, optimizer_1, self.basis, self.num_classes, 
            W, transition_matrix_group, idx_matrix_group, 
            func, model_save_dir=save_dir, epochs=self.matrix_epochs
        )
            # 清理极小值
        for i in range(basis_matrices.shape[0]):
            for j in range(basis_matrices.shape[1]):
                for k in range(basis_matrices.shape[2]):
                    if basis_matrices[i,j,k]<1e-6:
                        basis_matrices[i,j,k] = 0.0


        print("  [d] Detecting backdoor samples...")
        suspicious_samples, all_scores, all_matrices, all_anomaly_data = self.detect_backdoor_samples(
            W, basis_matrices, self.num_classes, 
            self.basis, self.detection_threshold
        )
        
        print("  [e] Evaluating detection results...")
        
        metrics = None
        if poison_indices is not None:
            poison_set = set(poison_indices)
            detected_set = set([s['idx'] for s in suspicious_samples])
            
            TP = len(poison_set & detected_set)
            FP = len(detected_set - poison_set)
            FN = len(poison_set - detected_set)
            TN = len(all_scores) - TP - FP - FN
            
            precision = TP / (TP + FP + 1e-8)
            recall = TP / (TP + FN + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'TN': TN
            }
            print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        return suspicious_samples, metrics, all_matrices, all_anomaly_data
    
    def get_features(self, model, dataloader):
        """
        使用hook从dataloader提取特征(倒数第二层)
        支持 PreActResNet 和 ResNeXt
        """
        features = []
        hook_features = []
        
        # 在最后的线性层之前注册hook
        handle = model.linear.register_forward_pre_hook(
            lambda module, input: hook_features.append(input[0].detach())
        )
        
        model.eval()
        with torch.no_grad():
            for i, (ins_data, ins_target) in enumerate(tqdm(dataloader, desc="Extracting features")):
                ins_data = ins_data.to(next(model.parameters()).device) # 保证数据和模型在同一设备
                hook_features.clear()
                _ = model(ins_data)
                
                # hook_features[0] 包含这个batch的所有特征
                batch_features = hook_features[0]
                for bid in range(len(ins_target)):
                    features.append(batch_features[bid].cpu().numpy())
        
        handle.remove()  # 移除hook

        return np.array(features)

    def get_probability(self, model, dataloader):
        """
        使用模型从dataloader提取预测概率
        """
        probabilities = []
        
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for data, _ in dataloader:
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
                
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())
        
        probabilities = np.vstack(probabilities)
        return probabilities
    
    def norm(self, T):
        row_sum = np.sum(T, 1)
        T_norm = T / row_sum
        return T_norm

    def estimate_matrix(self,logits_matrix, model_save_dir, exclude_indices=None):
        """
        估计锚点转移矩阵
        logits_matrix: (N, num_classes) 模型对所有样本的logits
        model_save_dir: 保存结果的路径
        return idx_matrix_group, transition_matrix_group
        idx_matrix_group: (num_classes, basis) 用于存放每个类别、每个 basis 下选中的“锚点”样本索引
        transition_matrix_group: (basis, num_classes, num_classes) 基转移矩阵组
        """
        transition_matrix_group = np.empty((self.basis, self.num_classes, self.num_classes))
        idx_matrix_group = np.empty((self.num_classes, self.basis), dtype=int)
        a = np.linspace(97, 99, args.basis)
        a = list(a)
        used_idx_per_class = [set() for _ in range(self.num_classes)]
        for i in range(len(a)):
            percentage = a[i]
            index = int(i)
            logits_matrix_ = copy.deepcopy(logits_matrix)
            # 过滤中毒样本
            if exclude_indices is not None:
                keep_mask = np.ones(logits_matrix_.shape[0], dtype=bool)
                keep_mask[exclude_indices] = False
                logits_matrix_ = logits_matrix_[keep_mask]
                # 记录剩余的原始索引
                kept_indices = np.arange(logits_matrix.shape[0])[keep_mask]
            else:
                kept_indices = np.arange(logits_matrix.shape[0])
            # fit 返回的是过滤后数据的行号，要映射回原始索引
            transition_matrix, idx = fit(logits_matrix_, self.num_classes, percentage, True)
            
            # 强制每个类别锚点不重复
            for c in range(self.num_classes):
                # 找到未被选中过的锚点
                candidates = [i for i in np.argsort(-logits_matrix_[:, c]) if kept_indices[i] not in used_idx_per_class[c]]
                if candidates:
                    chosen = candidates[0]
                    used_idx_per_class[c].add(kept_indices[chosen])
                    idx[c] = kept_indices[chosen]
                else:
                    # 如果所有样本都被选过，退而求其次，允许重复
                    idx[c] = kept_indices[np.argsort(-logits_matrix_[:, c])[0]]
            transition_matrix = self.norm(transition_matrix)
            idx_matrix_group[:, index] = np.array(idx)
            transition_matrix_group[index] = transition_matrix
        idx_group_save_dir = model_save_dir + '/' + 'idx_group.npy'
        group_save_dir = model_save_dir + '/' + 'T_group.npy'

        new_group = self.bias_transition_matrices(transition_matrix_group, target_class= 2, bias_strength=0.2)
        transition_matrix_group = new_group
        np.save(idx_group_save_dir, idx_matrix_group) 
        np.save(group_save_dir, transition_matrix_group) 
        return idx_matrix_group, transition_matrix_group

    def bias_transition_matrices(self,transition_matrix_group, target_class, bias_strength=0.2):
        """
        对转移矩阵组进行定向扰动，使其对目标类有更强的偏向。
        参数：
            transition_matrix_group: (basis, num_classes, num_classes) 原始转移矩阵组
            target_class: int，攻击目标类
            bias_strength: float，偏向强度（每行向目标类加多少概率）
        返回：
            new_transition_matrix_group: 偏向后的转移矩阵组
        """
        basis, num_classes, _ = transition_matrix_group.shape
        new_group = np.zeros_like(transition_matrix_group)
        for b in range(basis):
            T = transition_matrix_group[b].copy()
            for i in range(num_classes):
                # 对每一行，增加到目标类的概率
                T[i, target_class] += bias_strength
                # 重新归一化
                T[i, :] = T[i, :] / T[i, :].sum()
            new_group[b] = T
        return new_group

    def basis_matrix_optimize(self,model, optimizer, basis, num_classes, W_group, transition_matrix_group, idx_matrix_group, func, model_save_dir, epochs):
        """
        学习部分转移矩阵
        """
        basis_matrix_group = np.empty((basis, num_classes, num_classes))
        
        for i in range(num_classes):  

            model = init_params(model)
            for epoch in range(epochs):
                loss_total = 0.
                for j in range(basis):
                    class_1_idx = int(idx_matrix_group[i, j])
                    W = list(np.array(W_group[class_1_idx, :]))
                    T = torch.from_numpy(transition_matrix_group[j, i, :][:, np.newaxis]).float()
                    prediction = model(W[0], num_classes)
                    optimizer.zero_grad()
                    loss = func(prediction, T)
                    loss.backward()
                    optimizer.step()
                    loss_total += loss
                print(f"Class {i}, Epoch {epoch+1}/{epochs}, Loss: {loss_total:.6f}")
                if loss_total < 0.01:
                    break

            for x in range(basis):
                parameters = np.array(model.basis_matrix[x].weight.data)
        
                basis_matrix_group[x, i, :] = parameters
        A_save_dir = model_save_dir + '/' + 'A.npy'
        np.save(A_save_dir, basis_matrix_group)   
        return basis_matrix_group
    
    def matrix_combination(self,basis_matrix_group, W_group, idx, num_classes, basis):
        """
        构建实例转移矩阵
        """
        
        coefficient = W_group[idx, :]

        M = np.zeros((num_classes, num_classes))
        for i in range(basis):
            
            temp = float(coefficient[0, i]) * basis_matrix_group[i, :, :]
            M += temp
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i,j]<1e-6:
                    M[i,j] = 0.
        return M

class instance_matrix(defense):
    """Instance Matrix 检测方法主类"""
    
    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)
    
    @staticmethod
    def add_arguments(parser):
        """设置命令行参数"""
        parser.add_argument('--device', type=str, help='cuda, cpu')

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--yaml_path', type=str, default='./config/detection/instance_matrix/cifar10.yaml',
                           help='the path of yaml file')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--model', type=str, help='the model for classification')
        parser.add_argument('--batch_size', type=int)

        # Instance Matrix specific parameters
        parser.add_argument('--feature_dim', type=int, 
                           help='dimensions of feature vectors')
        parser.add_argument('--basis', type=int, 
                           help='basis matrix count')
        parser.add_argument('--nmf_iter', type=int, 
                           help='NMF iteration count')
        parser.add_argument('--nmf_threshold', type=float, 
                           help='NMF convergence threshold')
        parser.add_argument('--matrix_epochs', type=int, 
                           help='basis matrix learning epochs')
        parser.add_argument('--detection_threshold', type=float, 
                           help='detection threshold percentile')
        return parser
    
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/ins_pretrain/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
                
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model = model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    
    def set_devices(self):
        self.device = self.args.device
    
    def filtering(self):
        """主检测流程"""
        print("="*70)
        print("Instance Matrix Backdoor Detection")
        print("当前参数：", self.args.basis, self.args.nmf_iter, self.args.matrix_epochs)
        print("="*70)
        
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)  
        save_dir = self.args.save_path
              
        # 加载攻击结果
        print("\n[Stage 1/2] Loading attack result...")
        
        ## 重建模型
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
            model.eval()
        else:
            model.to(self.args.device)
            model.eval()
        
        ## 重建数据集 - 使用训练集进行检测
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        bd_train_dataset = self.result['bd_train'].wrapped_dataset
        poison_indices = np.where(np.array(bd_train_dataset.poison_indicator) == 1)[0]

        images_poison = []
        labels_poison = []
        for img, label,*other_info in bd_train_dataset:
            images_poison.append(img)
            labels_poison.append(label)
        
        bd_train_dataset = xy_iter(images_poison, labels_poison,transform=test_tran)
        bd_train_dataloader = DataLoader(bd_train_dataset,batch_size=self.args.batch_size, shuffle=False)

        # 保存中毒样本的索引
        all_labels = []
        for img, label, *other_info in bd_train_dataset:
            all_labels.append(label)
        all_labels = np.array(all_labels)
        poison_labels = all_labels[poison_indices]

        poison_idx_group = []
        for c in range(self.args.num_classes):
            idx_c = poison_indices[poison_labels == c]
            poison_idx_group.append(idx_c)
        poison_idx_group = np.array(poison_idx_group, dtype=object)
        poison_idx_save_path = os.path.join(save_dir, 'poison_idx_group.npy')
        np.save(poison_idx_save_path, poison_idx_group)
        print(f"Poison sample indices by class saved to: {poison_idx_save_path}")

        # 执行检测
        print("\n[Stage 2/2] Performing detection...")
        detector = InstanceMatrix(
            num_classes=self.args.num_classes,
            feature_dim=self.args.feature_dim,
            basis=self.args.basis,
            nmf_iter=self.args.nmf_iter,
            nmf_threshold=self.args.nmf_threshold,
            matrix_epochs=self.args.matrix_epochs,
            detection_threshold=self.args.detection_threshold
        )
        
        suspicious_samples, metrics, all_matrices ,all_anomaly_data = detector.detect(
            model, bd_train_dataloader, poison_indices, save_dir=save_dir
        )
        end = time.perf_counter()
        print(f"\n Total detection time: {end - start:.2f} seconds")
        # 显示结果
        print(f"\n Detected {len(suspicious_samples)} suspicious samples")
        print(f"   Detection rate: {len(suspicious_samples)/len(bd_train_dataloader.dataset)*100:.2f}%")
        
        # 显示top 10可疑样本
        print("\n  Top 10 suspicious samples:")
        for i, sample in enumerate(suspicious_samples[:10]):
            print(f"    #{i+1} Sample {sample['idx']:5d}: "
                  f"score={sample['score']:.4f}, "
                  f"diag_dom={sample['diagonal_dominance']:.3f}, "
                  f"entropy={sample['entropy']:.3f}, "
                  f"max_off_diag={sample['max_off_diagonal']:.3f}")
        
        # 评估指标
        if metrics:
            print(f"\n Detection Metrics:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1']:.4f}")
            print(f"    TP: {metrics['TP']}, FP: {metrics['FP']}, "
                  f"FN: {metrics['FN']}, TN: {metrics['TN']}")
        
        param_suffix = f"basis{self.args.basis}_nmf{self.args.nmf_iter}_epoch{self.args.matrix_epochs}"

        # 保存所有样本的检测结果
        poison_set = set(poison_indices)
        for item in all_anomaly_data:
            item['is_poison'] = int(item['idx'] in poison_set)

        all_samples_save_path = os.path.join(save_dir, f'instance_matrix_all_samples_{param_suffix}.csv')
        all_samples_df = pd.DataFrame(all_anomaly_data)
        all_samples_df.to_csv(all_samples_save_path, index=False)
        print(f" All samples' detection features saved to: {all_samples_save_path}")
        
        # 保存检测结果
        detec_save_path = os.path.join(save_dir, f'instance_matrix_detection_{param_suffix}.csv')
        result_df = pd.DataFrame(suspicious_samples)
        result_df.to_csv(detec_save_path, index=False)
        print(f"\n Detection results saved to: {detec_save_path}")
        
        # 保存所有样本的实例转移矩阵
        matrices_save_path = os.path.join(save_dir, f'instance_matrices_{param_suffix}.pickle')
        with open(matrices_save_path, 'wb') as f:
            pickle.dump(all_matrices, f)
        print(f" Instance matrices saved to: {matrices_save_path}")
        
        # 保存评估结果
        if metrics:
            eval_save_path = os.path.join(save_dir, f'instance_matrix_evaluation_{param_suffix}.csv')
            eval_df = pd.DataFrame([{
                'method': 'instance_matrix',
                'num_detected': len(suspicious_samples),
                'detection_rate': len(suspicious_samples)/len(bd_train_dataloader.dataset),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'TN': metrics['TN']
            }])
            eval_df.to_csv(eval_save_path, index=False)
            print(f" Evaluation metrics saved to: {eval_save_path}")
        
        # 保存检测配置信息
        config_save_path = os.path.join(save_dir, f'detection_config_{param_suffix}.yaml')
        config_dict = {
            'detection_method': 'instance_matrix',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': self.args.dataset,
            'model': self.args.model,
            'num_classes': self.args.num_classes,
            'result_file': self.args.result_file,
            'parameters': {
                'feature_dim': self.args.feature_dim,
                'basis': self.args.basis,
                'nmf_iter': self.args.nmf_iter,
                'nmf_threshold': self.args.nmf_threshold,
                'matrix_epochs': self.args.matrix_epochs,
                'detection_threshold': self.args.detection_threshold,
                'batch_size': self.args.batch_size,
                'random_seed': self.args.random_seed
            }
        }
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f" Detection config saved to: {config_save_path}")
        
        print("\n" + "="*70)
        print("Detection completed!")
        print(f"All results saved in: {save_dir}")
        print("="*70)
    
    def detection(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()
        return result 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    instance_matrix.add_arguments(parser)
    args = parser.parse_args()
    ins_method = instance_matrix(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = ins_method.detection(args.result_file)