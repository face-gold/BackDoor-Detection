import argparse
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import yaml
import logging
from pprint import pformat
import csv
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append('../')
sys.path.append(os.getcwd())

# 复用 BackdoorBench 的基础组件
from defense.base import defense
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result
from utils.log_assist import get_git_info
from torch.utils.data import DataLoader

# =============================================================================
# 核心算法类: LabelFlipDetectorAlgo
# =============================================================================

class LabelFlipDetectorAlgo:
    def __init__(self, args, model, train_loader, num_classes, num_samples, label_true):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.device = args.device
        self.label_true = label_true
        
        # 记录每个样本在关键节点的 Loss
        # shape: [N], 初始化为 -1
        self.loss_at_flip_end = torch.zeros(num_samples).to(self.device) - 1
        self.loss_at_recovery_start = torch.zeros(num_samples).to(self.device) - 1
        
        # 记录平均 Loss 用于画图
        self.clean_loss_history = []
        self.poison_loss_history = []
        
        self.all_epoch_loss = []
        
        self.flip_start_epoch = args.flip_start_epoch
        self.recovery_start_epoch = args.recovery_start_epoch
        self.total_epochs = args.epochs

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        # 使用 Cosine 调度，覆盖整个训练周期
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_epochs)
        
        # 关键：使用 reduction='none' 以便获取每个样本的 Loss
        criterion = nn.CrossEntropyLoss(reduction='none')

        print(f"[*] Starting Label Flip Probe Detection...")
        print(f"    - Normal Phase: 0 -> {self.flip_start_epoch}")
        print(f"    - Flip Phase  : {self.flip_start_epoch} -> {self.recovery_start_epoch}")
        print(f"    - Recovery    : {self.recovery_start_epoch} -> {self.total_epochs}")

        for epoch in range(self.total_epochs):
            self.model.train()
            
            # 标记当前阶段
            is_flip_phase = (epoch >= self.flip_start_epoch) and (epoch < self.recovery_start_epoch)
            # is_recovery_phase = (epoch >= self.recovery_start_epoch)
            
            epoch_loss_clean = []
            epoch_loss_poison = []
            
            epoch_loss_all = np.zeros(self.num_samples)
            
            for batch_idx, (data, labels, index, _, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                index = index.to(self.device)

                # ==========================
                # 1. 实施标签翻转逻辑
                # ==========================
                current_labels = labels.clone()
                if is_flip_phase:
                    # 循环移位翻转: 0->1, 1->2, ..., 9->0
                    current_labels = (current_labels + 1) % self.num_classes
                
                # ==========================
                # 2. 正常训练步
                # ==========================
                optimizer.zero_grad()
                output = self.model(data)
                
                # 计算每个样本的 Loss
                loss_per_sample = criterion(output, current_labels)
                loss = loss_per_sample.mean()
                
                # 记录每个样本的loss
                idx_np = index.cpu().numpy()
                loss_np = loss_per_sample.detach().cpu().numpy()
                epoch_loss_all[idx_np] = loss_np
                
                loss.backward()
                optimizer.step()
                
                # ==========================
                # 3. 记录关键数据
                # ==========================
                with torch.no_grad():
                    # 分别记录干净和中毒样本的 Loss 用于画图
                    # 注意：这里需要 cpu() 转换，比较慢，实际使用可优化
                    idx_np = index.cpu().numpy()
                    loss_np = loss_per_sample.detach().cpu().numpy()
                    
                    # 假设我们有 ground truth (label_true) 用于画图验证，实际检测时不使用
                    for i, idx_val in enumerate(idx_np):
                        if self.label_true[idx_val] == 1: # Poison
                            epoch_loss_poison.append(loss_np[i])
                        else:
                            epoch_loss_clean.append(loss_np[i])

                    # [关键检测点 1] 翻转阶段的最后一刻 (在进入恢复阶段前的最后一个 epoch)
                    if epoch == self.recovery_start_epoch - 1:
                        self.loss_at_flip_end[index] = loss_per_sample.detach()
                    
                    # [关键检测点 2] 恢复阶段的第一刻 (刚把标签改回来训练的第一个 epoch)
                    if epoch == self.recovery_start_epoch:
                        self.loss_at_recovery_start[index] = loss_per_sample.detach()

            scheduler.step()
            
            self.all_epoch_loss.append(epoch_loss_all)
            
            # 记录本 Epoch 平均值
            avg_clean = np.mean(epoch_loss_clean) if len(epoch_loss_clean)>0 else 0
            avg_poison = np.mean(epoch_loss_poison) if len(epoch_loss_poison)>0 else 0
            self.clean_loss_history.append(avg_clean)
            self.poison_loss_history.append(avg_poison)
            
            # 打印日志
            phase_str = "NORMAL"
            if epoch >= self.flip_start_epoch and epoch < self.recovery_start_epoch:
                phase_str = "FLIP"
            elif epoch >= self.recovery_start_epoch:
                phase_str = "RECOVER"

            print(f"Epoch {epoch:2d} | Phase: {phase_str:7s} | Loss Clean: {avg_clean:.4f} | Loss Poison: {avg_poison:.4f}")

        # 训练结束，绘制 Loss 曲线验证想法
        self.plot_loss_dynamics()
        
        all_epoch_loss_np = np.stack(self.all_epoch_loss, axis=0) # shape: [epochs, N]
        records = []
        for epoch in range(self.total_epochs):
            for idx in range(self.num_samples):
                records.append({
                    'epoch': epoch,
                    'index': idx,
                    'loss': all_epoch_loss_np[epoch, idx],
                    'is_poison': int(self.label_true[idx])
                })
        df = pd.DataFrame(records)
        csv_path = os.path.join(self.args.save_path, 'all_epoch_loss.csv')
        df.to_csv(csv_path, index=False)
        print(f"[*] All epoch loss saved to {csv_path}")

    def get_suspect_indices(self):
        """
        计算检测得分并返回嫌疑人索引
        """
        # 核心指标：恢复阶段第1个Epoch的 Loss 值。
        # 原理：中毒样本记忆恢复极快，Loss 会瞬间接近 0；干净样本恢复慢，Loss 较高。
        # Score 越大代表越有可能是后门 -> 取 -Loss
        
        loss_recover = self.loss_at_recovery_start.cpu().numpy()
        scores = loss_recover 
        
        return scores

    def plot_loss_dynamics(self):
        plt.figure(figsize=(10, 6))
        epochs = range(self.total_epochs)
        plt.plot(epochs, self.clean_loss_history, label='Clean Samples', color='blue', linewidth=2)
        plt.plot(epochs, self.poison_loss_history, label='Poison Samples', color='red', linewidth=2, linestyle='--')
        
        # 画出阶段分界线
        plt.axvline(x=self.flip_start_epoch, color='gray', linestyle=':', label='Flip Start')
        plt.axvline(x=self.recovery_start_epoch, color='gray', linestyle=':', label='Recovery Start')
        
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss Dynamics under Label Flipping')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_file = os.path.join(self.args.save_path, 'flip_loss_dynamics.png')
        plt.savefig(save_file)
        print(f"[*] Loss dynamics plot saved to {save_file}")
        plt.close()

# =============================================================================
# 包装器类 (适配 BackdoorBench 接口)
# =============================================================================
class LabelFlipProbe(defense):
    def __init__(self, args):
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
        
        if 'result_file' in args.__dict__ and args.result_file is not None:
            self.set_result(args.result_file)

    def add_arguments(parser):
        # 基础参数
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])
        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        # 训练超参
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument("--num_workers", type=float, default=4)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--model', type=str, help='preactresnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/flip/cifar10.yaml", help='the path of yaml')
        
        # Label Flip Probe 特有参数
        parser.add_argument('--epochs', type=int, help='Total training epochs')
        parser.add_argument('--flip_start_epoch', type=int, help='Epoch to start label flipping (e.g. 20)')
        parser.add_argument('--recovery_start_epoch', type=int, help='Epoch to stop flipping and recover labels (e.g. 30)')
        
        parser.add_argument('--pratio', type=float, default=0.1, help='Estimated poison ratio for filtering')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/flip/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path) 
        self.args.save_path = save_path
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def detection(self, result_file):
        self.set_result(result_file)
        fix_random(self.args.random_seed)
        
        # 1. 准备数据
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.to(self.args.device)
        
        inner_dataset = self.result['bd_train'].wrapped_dataset # 原始未包装数据集
        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=True)
        bd_train_dataset = dataset_wrapper_with_transform(inner_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        # 获取 Ground Truth 用于验证
        pindex = np.where(np.array(inner_dataset.poison_indicator) == 1)[0]
        label_true = np.zeros(len(bd_train_dataset))
        label_true[pindex] = 1
        
        train_loader = DataLoader(bd_train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        
        # 2. 运行探针算法 (传入 args，内部会自动读取 epoch 参数)
        detector = LabelFlipDetectorAlgo(self.args, model, train_loader, self.args.num_classes, len(bd_train_dataset), label_true)
        detector.train()
        
        # 3. 获取得分并筛选
        outlier_scores = detector.get_suspect_indices()
        
        # === 调试代码 Start ===
        print("-" * 30)
        print("DEBUG INFO:")

        # 1. 检查是否有数据没被记录 (是否还有 -1 的初始值)
        if (outlier_scores == -1).any():
            print("[WARNING] detected un-updated scores (-1)!")
            
        # 2. 打印中毒和干净样本的平均得分
        poison_scores = outlier_scores[label_true == 1]
        clean_scores = outlier_scores[label_true == 0]

        print(f"Mean Poison Score: {np.mean(poison_scores):.4f}")
        print(f"Mean Clean Score:  {np.mean(clean_scores):.4f}")

        # 3. 简单的手动判断
        if np.mean(poison_scores) > np.mean(clean_scores):
            print("Trend is Correct: Poison > Clean")
        else:
            print("Trend is WRONG: Clean > Poison")
        print("-" * 30)
        # === 调试代码 End ===
        
        # 计算 AUC
        fpr, tpr, thresholds = metrics.roc_curve(label_true, outlier_scores)
        auc = metrics.auc(fpr, tpr)
        print(f"Detection AUC: {auc:.4f}")
        
        # 筛选 Top-K
        num_poison_expected = int(len(bd_train_dataset) * self.args.pratio)
        suspect_index = np.argsort(outlier_scores)[-num_poison_expected:]
        
        # 保存结果
        with open(self.args.save_path + '/detection_info.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['LabelFlipProbe', 'AUC', auc])
            
        return suspect_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    LabelFlipProbe.add_arguments(parser)
    
    args = parser.parse_args()
    
    method = LabelFlipProbe(args)
    
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'

    method.detection(args.result_file)