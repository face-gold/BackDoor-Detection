import argparse
import os, sys
import numpy as np
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

from defense.base import defense
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result
from utils.log_assist import get_git_info
from torch.utils.data import DataLoader


from detection_pretrain.coinnet.coinnet_utils import generate_crowd_labels

# =============================================================================
# Part 1: COINNet 核心算法类 
# =============================================================================

class COINNetModel(nn.Module):
    """
    COINNet 的可学习模型，包含骨干网络、混淆矩阵 P0 和 实例扰动 E
    """
    def __init__(self, backbone, num_classes, num_samples, num_annotators=3):
        super(COINNetModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 1. 混淆矩阵 P0 (Stack of Identity matrices)
        self.P0 = nn.Parameter(torch.stack([torch.eye(num_classes) for _ in range(num_annotators)]))
        
        # 2. 实例相关扰动 E (N x M x K), 初始化为 0
        self.E = nn.Parameter(torch.zeros(num_samples, num_annotators, num_classes))
        
        self.P0_normalize = nn.Softmax(dim=1) 

    def E_normalize(self, x):
        return x - x.mean(dim=-1, keepdim=True)

    def forward(self, x, indices):
        # 1. Backbone logits -> softmax
        logits = self.backbone(x)
        f_x = F.softmax(logits, dim=1) # [Batch, K]
        
        # 2. P0 logic
        P0 = self.P0_normalize(self.P0) # [M, K, K]
        # y = P0 * f(x) -> [Batch, M, K]
        y = torch.einsum('mkj,bj -> bmk', P0, f_x)
        
        # 3. E logic
        batch_E = self.E[indices] 
        e = self.E_normalize(batch_E)
        
        # 4. Combine
        y = y + e
        
        # 5. Numerical Stability (HACKING from original code)
        y = torch.clamp(y, min=1e-10, max=1.0 - 1e-10)
        y = y / y.sum(dim=-1, keepdim=True)
        
        return y, e, f_x

    def get_e_global(self):
        return self.E_normalize(self.E)

class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        """
        output: 模型预测的概率分布 [Batch, K] (这里对应 Af_x)
        target: 众包标签的 One-Hot 编码 [Batch, K] (或者软标签)
        """
        # 1. 计算联合分布矩阵 U (近似混淆矩阵 *先验)
        # U = output.T @ target
        # 形状: [K, Batch] @ [Batch, K] -> [K, K]
        U = torch.mm(output.t(), target)
        
        # 2. 归一化 (可选，但在Batch Size变化时有帮助)
        U = U / output.shape[0] 

        # 3. 计算行列式的绝对值
        # 为了数值稳定性，在对角线加上微小的扰动
        loss = -torch.log(torch.abs(torch.det(U)) + 1e-6)
        
        return loss

class COINNetAlgo:
    """
    COINNet 算法逻辑封装
    """
    def __init__(self, args, model, train_loader, crowd_labels, num_classes, num_samples, label_true):
        self.args = args
        self.model = model # 这里的 model 是骨干网络
        self.train_loader = train_loader
        self.crowd_labels = crowd_labels
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.device = args.device
        self.label_true = label_true # Ground truth labels for evaluation [0, 0, 1, 0...]
        
        # 初始化 COINNet 框架
        self.coin_model = COINNetModel(self.model, num_classes, num_samples, args.num_annotators)
        self.coin_model = self.coin_model.to(self.device)
        
        # 记录每个 Epoch 各个类别的预测数量
        self.class_dist_history = [] 

        # Init metric logger file
        self.metric_log_file = os.path.join(self.args.save_path, 'epoch_metrics.txt')
        with open(self.metric_log_file, 'w') as f:
            f.write("Epoch\tTN\tFP\tFN\tTP\tTPR\tFPR\tAUC\tLoss\n")

    def train(self):
        optimizer = optim.Adam(self.coin_model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, epochs=self.args.epochs, steps_per_epoch=len(self.train_loader)
        )
        criterion = nn.NLLLoss(ignore_index=-1, reduction='mean')

        self.coin_model.train()
        print(f"[*] Training COINNet for {self.args.epochs} epochs...")
        
        for epoch in range(self.args.epochs):
            batch_loss = []
            epoch_preds = [] # 收集本 Epoch 所有样本的预测
            for batch_data in self.train_loader:
                # 适配 BackdoorBench 5元组
                if len(batch_data) == 5:
                    data, _, index, _, _ = batch_data
                elif len(batch_data) == 3:
                    data, _, index = batch_data
                else:
                    continue # Skip invalid data

                data = data.to(self.device)
                index = index.to(self.device)
                batch_targets = self.crowd_labels[index]

                optimizer.zero_grad()
                
                # Forward
                Af_x, e_batch, f_x = self.coin_model(data, index)
                
                # 获取骨干网络对每个样本的预测类别
                preds = torch.argmax(f_x, dim=1).detach().cpu().numpy()
                epoch_preds.append(preds)

                # 1. CE Loss
                loss_ce = criterion(Af_x.reshape(-1, self.num_classes).log(), batch_targets.view(-1).long())
                if loss_ce > 100: loss_ce = torch.tensor(0.0).to(self.device)

                # 2. Volume Loss （Type 'f'使用特征矩阵的体积正则——对特征多样性进行约束——效果好些
                HH = torch.mm(f_x.t(), f_x)
                logdet = -torch.log(torch.linalg.det(HH) + 1e-6)
                if (torch.isnan(logdet) or torch.isinf(logdet) or logdet < -100):
                    logdet = torch.tensor(0.0).to(self.device)

                # 3. Sparsity Loss
                err_sq = (e_batch ** 2).sum(dim=(1, 2)) + 1e-10
                loss_sparse = (err_sq ** 0.2).mean()

                loss = loss_ce + self.args.lam1 * logdet + self.args.lam2 * loss_sparse
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_loss.append(loss.item())
            
            # --- [每个 Epoch 结束时记录数据] ---
            # 拼接所有 batch 的预测
            all_preds = np.concatenate(epoch_preds)
            # 统计当前 Epoch 每个类别的数量
            # bincount 统计 0~K-1 出现的次数，minlength 保证即使某类没出现也为 0
            counts = np.bincount(all_preds, minlength=self.num_classes)
            self.class_dist_history.append(counts)

            mean_loss = np.mean(batch_loss)
            
            # Evaluate Metrics at the end of EACH epoch
            self.evaluate_epoch(epoch, mean_loss)

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1} | Loss: {mean_loss:.4f}")
            if mean_loss < 1e-4:
                print(f"    Early stopping at epoch {epoch+1} with loss {mean_loss:.6f}")
                break
            
        
        print("[*] Training finished. Generating plots...")
        self.plot_class_trends()
        # [NEW] Call the metric plotting function
        self.plot_training_metrics()

    # Helper functions for metric calculation
    def cal(self, true, pred):
        TN, FP, FN, TP = metrics.confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN) if (TP+FN) > 0 else 0
        FPR = FP/(FP+TN) if (FP+TN) > 0 else 0
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc
    
    def get_outlier_scores(self):
        """
        计算每个样本的离群得分 (E 的能量)
        """
        final_E = self.coin_model.get_e_global().detach() # [N, M, K]
        # err = (E**2).sum((1, 2))
        err_scores = (final_E ** 2).sum(dim=(1, 2)).cpu().numpy()
        return err_scores

    def evaluate_epoch(self, epoch, loss):
        # 1. Calculate Outlier Scores
        outlier_scores = self.get_outlier_scores()
        
        # 2. Calculate AUC
        label_true = self.label_true
        label_pred = outlier_scores
        
        try:
            fpr, tpr, thresholds = metrics.roc_curve(label_true, label_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
        except Exception as e:
            auc = 0.0
            print(f"[!] AUC calculation failed at epoch {epoch}: {e}")

        # 3. Calculate Threshold metrics (TN, FP, FN, TP, TPR, FPR)
        # Assume filtering top-K based on pratio
        num_poison_expected = int(len(label_true) * self.args.pratio)
        suspect_index = np.argsort(label_pred)[-num_poison_expected:]
        
        if len(suspect_index) == 0:
            tn = len(label_true) - np.sum(label_true)
            fp = np.sum(label_true)
            fn = 0
            tp = 0
            TPR, FPR = 0, 0
        else:
            findex = np.zeros(len(label_true))
            findex[suspect_index] = 1
            
            try:
                tn, fp, fn, tp = self.cal(label_true, findex)
                TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)
            except ValueError:
                # Handle edge cases where confusion matrix shape might be wrong (e.g. all 0 or all 1)
                tn, fp, fn, tp = 0, 0, 0, 0
                TPR, FPR = 0, 0

        # 4. Save to txt file
        with open(self.metric_log_file, 'a') as f:
            f.write(f"{epoch+1}\t{tn}\t{fp}\t{fn}\t{tp}\t{TPR:.4f}\t{FPR:.4f}\t{auc:.4f}\t{loss:.4f}\n")

    # [NEW] Function to plot training metrics from log file
    def plot_training_metrics(self):
        metric_file = self.metric_log_file
        if not os.path.exists(metric_file):
            print(f"[!] Metric file not found: {metric_file}")
            return

        epochs = []
        tns, fps, fns, tps = [], [], [], []
        tprs, fprs = [], []
        aucs, losses = [], []

        try:
            with open(metric_file, 'r') as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) < 9: continue
                    epochs.append(int(parts[0]))
                    tns.append(int(parts[1]))
                    fps.append(int(parts[2]))
                    fns.append(int(parts[3]))
                    tps.append(int(parts[4]))
                    tprs.append(float(parts[5]))
                    fprs.append(float(parts[6]))
                    aucs.append(float(parts[7]))
                    losses.append(float(parts[8]))
        except Exception as e:
            print(f"[!] Error reading metric file: {e}")
            return

        save_dir = self.args.save_path
        
        # 1. AUC and Loss Plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('AUC', color=color)
        ax1.plot(epochs, aucs, label='AUC', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
        ax2.plot(epochs, losses, label='Loss', color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Detection AUC and Training Loss over Epochs')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(os.path.join(save_dir, 'metric_auc_loss.png'), dpi=150)
        plt.close()

        # 2. TPR and FPR Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, tprs, label='TPR (Recall)', color='green')
        plt.plot(epochs, fprs, label='FPR', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.ylim([0, 1.1])
        plt.title('TPR and FPR over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'metric_tpr_fpr.png'), dpi=150)
        plt.close()
        
        # 3. Confusion Matrix Elements Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, tps, label='TP (True Poison)', linestyle='-')
        plt.plot(epochs, fps, label='FP (False Poison)', linestyle='--')
        plt.plot(epochs, fns, label='FN (Missed Poison)', linestyle=':')
        # TN might be too large compared to others, usually we care about TP/FP/FN in detection
        # But plotting it for completeness
        # plt.plot(epochs, tns, label='TN (True Clean)', alpha=0.3) 
        
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.title('Confusion Matrix Breakdown (TP/FP/FN)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'metric_confusion_matrix.png'), dpi=150)
        plt.close()
        
        print(f"[*] Metric plots saved to {save_dir}")

    def plot_class_trends(self):
        """
        绘制类别预测变化图并保存
        """
        # 1. 准备数据
        history = np.array(self.class_dist_history)
        
        if len(history) == 0:
            print("[!] No history data to plot.")
            return
            
        epochs = np.arange(1, len(history) + 1)
        save_dir = self.args.save_path

        # ---------------------------------------------------------
        # 图表 1: 折线图 (Line Plot)
        # ---------------------------------------------------------
        plt.figure(figsize=(12, 6))
        
        for k in range(self.num_classes):
            # 现在 epochs 和 history[:, k] 的长度一定相等了
            plt.plot(epochs, history[:, k], label=f'Class {k}', linewidth=1.5)
            
        plt.title('Prediction Count per Class over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Samples Predicted')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        line_plot_path = os.path.join(save_dir, 'class_prediction_trend.png')
        plt.savefig(line_plot_path, dpi=150)
        plt.close()
        
        # ---------------------------------------------------------
        # 图表 2: 热力图 (Heatmap)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 8))
        
        plt.imshow(history, aspect='auto', cmap='viridis', origin='upper', interpolation='nearest')
        
        plt.colorbar(label='Sample Count')
        plt.title('Class Prediction Heatmap (Epoch vs Class)')
        plt.xlabel('Class ID')
        plt.ylabel('Epoch (0 = start)')
        plt.xticks(np.arange(self.num_classes), np.arange(self.num_classes))
        
        plt.tight_layout()
        heatmap_path = os.path.join(save_dir, 'class_prediction_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        print(f"[*] Plots saved to:\n    - {line_plot_path}\n    - {heatmap_path}")
    

# =============================================================================
# Part 2: BackdoorBench 防御基类实现
# =============================================================================

class coinnet(defense):

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

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
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
    
        parser.add_argument('--epochs', type=int,help='COINNet training epochs')
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument("--num_workers", type=float, default=4)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--model', type=str, help='preactresnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/coinnet/cifar10.yaml", help='the path of yaml')
        
        # COINNet 特有参数
        parser.add_argument('--num_annotators', type=int, help='Number of simulated annotators')
        parser.add_argument('--lam1', type=float, help='Weight for Volume Minimization')
        parser.add_argument('--lam2', type=float, help='Weight for Sparsity')
        parser.add_argument('--pratio', type=float, default=0.1, help='Estimated poison ratio for filtering')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/coinnet_pretrain/'
        if not (os.path.exists(save_path)):
                os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'detection_info/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
                
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # Ensure we don't add handlers multiple times
        if not logger.handlers:
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

    def cal(self, true, pred):
        TN, FP, FN, TP = metrics.confusion_matrix(true, pred).ravel()
        return TN, FP, FN, TP 
    
    def metrix(self, TN, FP, FN, TP):
        TPR = TP/(TP+FN) if (TP+FN) > 0 else 0
        FPR = FP/(FP+TN) if (FP+TN) > 0 else 0
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        acc = (TP+TN)/(TN+FP+FN+TP)
        return TPR, FPR, precision, acc

    def filtering(self):
        start = time.perf_counter()
        self.set_devices()
        fix_random(self.args.random_seed)

        # ---------------------------------------------
        # 1. 准备模型和数据
        # ---------------------------------------------
        # 加载攻击后保存的骨干模型 (Backdoor Model)
        model = generate_cls_model(self.args.model, self.args.num_classes)
        model.to(self.args.device)
        
        # 提取数据
        inner_dataset = self.result['bd_train'].wrapped_dataset
        
        # 获取数据变换
        train_tran = get_transform(
            self.args.dataset, 
            *([self.args.input_height, self.args.input_width]), 
            train=True
        )
        # 重新包装数据集 (加上 Transform)
        bd_train_dataset = dataset_wrapper_with_transform(
            inner_dataset,
            wrap_img_transform=train_tran,
            wrap_label_transform=None
        )
        
        # 获取真实的投毒索引 (Ground Truth) 用于计算 AUC
        pindex = np.where(np.array(inner_dataset.poison_indicator) == 1)[0]
        torch.save(torch.tensor(pindex), self.args.save_path + '/pindex.pt')
        
        # [MODIFIED] Construct Ground Truth Label Array for Online Evaluation
        num_samples = len(bd_train_dataset)
        label_true = np.zeros(num_samples)
        label_true[pindex] = 1 # 1 = poison, 0 = clean

        # 构造 DataLoader
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
        

        # ---------------------------------------------
        # 2. COINNet 流程 Part A: 生成众包标签
        # ---------------------------------------------
        # 定义保存路径
        npy_path = os.path.join(self.args.save_path, 'crowd_labels.npy')
        txt_path = os.path.join(self.args.save_path, 'crowd_labels.txt')
        
        # 逻辑：优先检查 .npy 文件是否存在
        if os.path.exists(npy_path):
            logging.info(f"Found existing crowd labels at {npy_path}")
            logging.info("Skipping generation and loading directly...")
            
            crowd_labels_np = np.load(npy_path)
            crowd_labels = torch.from_numpy(crowd_labels_np).long()
            
        else:
            logging.info("Step 1: Generating Crowd Labels (No cache found)...")
            
            crowd_labels = generate_crowd_labels(
                train_loader, 
                self.args.device, 
                self.args.num_classes, 
                num_samples, 
                self.args.num_annotators
            )
            
            if isinstance(crowd_labels, torch.Tensor):
                crowd_labels_np = crowd_labels.cpu().numpy()
            else:
                crowd_labels_np = crowd_labels
                crowd_labels = torch.from_numpy(crowd_labels).long() 

            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)

            # 4. 保存
            np.save(npy_path, crowd_labels_np)
            np.savetxt(txt_path, crowd_labels_np, fmt='%d', delimiter='\t')
            logging.info(f"Labels saved to:\n  - {npy_path}\n  - {txt_path}")

        # 统一移动到设备
        crowd_labels = crowd_labels.to(self.args.device)

        # ---------------------------------------------
        # 3. COINNet 流程 Part B: 训练框架
        # ---------------------------------------------
        logging.info("Step 2: Training COINNet Framework...")
        # [MODIFIED] Pass label_true to COINNetAlgo
        worker = COINNetAlgo(self.args, model, train_loader, crowd_labels, self.args.num_classes, num_samples, label_true)
        worker.train()
        
        # 保存训练好的转移矩阵到txt文件中
        model_instance = worker.coin_model
        nominal_matrix = model_instance.P0_normalize(model_instance.P0)
        nominal_matrix_np = nominal_matrix.detach().cpu().numpy() # Shape: [M, Noisy, True]

        logging.info(f"Learned Nominal Matrix Shape: {nominal_matrix_np.shape}")
        
        txt_save_path = os.path.join(self.args.save_path, 'nominal_matrices.txt')
        
        try:
            with open(txt_save_path, 'w') as f:
                f.write(f"Learned Nominal Matrices (M={nominal_matrix_np.shape[0]}, K={nominal_matrix_np.shape[1]})\n")
                # 列是真实标签，行是预测(噪声)标签
                f.write("Format: Column Stochastic Matrix (Columns sum to 1)\n")
                f.write("ROWS (Down): Predicted/Noisy Label (k)\n")
                f.write("COLS (Across): True/Ground-Truth Label (j)\n")
                f.write("Value P_kj = P(Noisy=k | True=j)\n")
                f.write("=" * 60 + "\n\n")

                for m in range(nominal_matrix_np.shape[0]):
                    f.write(f"Annotator #{m} Confusion Matrix:\n")
                    f.write("-" * 30 + "\n")
                    # 保存矩阵
                    np.savetxt(f, nominal_matrix_np[m], fmt='%.4f', delimiter='\t')
                    f.write("\n" + "=" * 60 + "\n\n")
            
            logging.info(f"Successfully saved confusion matrices to: {txt_save_path}")
            
        except Exception as e:
            logging.error(f"Failed to save nominal matrices: {e}")

        # ---------------------------------------------
        # 4. COINNet 流程 Part C: 检测与评分
        # ---------------------------------------------
        logging.info("Step 3: Detecting Outliers (Final Evaluation)...")
        # 获取离群得分 (err_scores)，分数越高越可能是后门
        outlier_scores = worker.get_outlier_scores()
        
        # 保存得分
        torch.save(torch.tensor(outlier_scores), self.args.save_path + '/outlier_scores.pt')

        # 计算 ROC/AUC
        # label_true 已经在上面定义了
        
        # COINNet 中 E 的能量越大，表示越是离群值 (Poison)
        label_pred = outlier_scores 
        
        # ====== 保存三元组到CSV ======
        indices = np.arange(num_samples)
        with open(self.args.save_path + '/coinnet_scores.csv', 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['index', 'is_poison', 'outlier_score'])
            for idx, is_p, score in zip(indices, label_true, outlier_scores):
                writer.writerow([idx, int(is_p), float(score)])
        # ====== 保存三元组到CSV结束 ======
        
        fpr, tpr, thresholds = metrics.roc_curve(label_true, label_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        logging.info(f"Detection AUC: {auc:.4f}")

        # 确定筛选阈值 (Top-K filtering)
        # 假设已知 pratio
        num_poison_expected = int(num_samples * self.args.pratio)
        # 获取得分最高的 K 个样本的索引
        suspect_index = np.argsort(label_pred)[-num_poison_expected:]

        # ---------------------------------------------
        # 5. 记录 Metrics (CSV)
        # ---------------------------------------------
        if len(suspect_index) == 0:
            tn = len(label_true) - np.sum(label_true)
            fp = np.sum(label_true)
            fn = 0
            tp = 0
            TPR, FPR, auc = 0, 0, auc
        else:
            findex = np.zeros(num_samples)
            findex[suspect_index] = 1
            tn, fp, fn, tp = self.cal(label_true, findex)
            TPR, FPR, precision, acc = self.metrix(tn, fp, fn, tp)

        end = time.perf_counter()
        time_minute = (end-start)/60

        f = open(self.args.save_path + '/detection_info.csv', 'a', encoding='utf-8')
        csv_write = csv.writer(f)
        csv_write.writerow(['record', 'TN','FP','FN','TP','TPR','FPR', 'AUC', 'target', 'Time(min)'])
        csv_write.writerow([self.args.result_file, tn, fp, fn, tp, TPR, FPR, auc, 'None', time_minute])
        f.close()
        
        logging.info(f"Finished. TPR: {TPR:.4f}, FPR: {FPR:.4f}, AUC: {auc:.4f}")

        # 返回被认为是干净的样本索引 (BackdoorBench 标准接口可能需要这个，也可能只需要 suspect)
        # 这里返回 suspect_index (被检测为有毒的索引)
        return suspect_index

    def detection(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.filtering()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    coinnet.add_arguments(parser)
    args = parser.parse_args()
    
    coinnet_method = coinnet(args)
    
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
        
    result = coinnet_method.detection(args.result_file)