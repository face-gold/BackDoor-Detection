# 修改网络为preactResnet，和backdoorbench的训练保持一致
import argparse
import os
import sys
import yaml
import time
import logging
from pprint import pformat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


sys.path.append('../')
sys.path.append(os.getcwd())

from utils.aggregate_block.fix_random import fix_random
from defense.base import defense
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_transform
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

from bltm_detect import resnet 
from bltm_detect import resnet_bayes
from bltm_detect import preact_resnet
from bltm_detect import preact_resnet_T

class BLTMTrainer(defense):
    
    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults
        args.terminal_info = sys.argv
        
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        
        self.args = args

        if 'result_file' in args.__dict__ and args.result_file is not None:
            self.set_result(args.result_file)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--checkpoint_save', type=str, help='location to save models')
        parser.add_argument('--log', type=str, help='location of log')
        parser.add_argument("--dataset_path", type=str, help='dataset location')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='result folder name')
    
        # 训练参数
        parser.add_argument('--epochs', type=int, help='epochs for T-Net training')
        parser.add_argument('--warmup_epochs', type=int, help='epochs for classifier warm-up')
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--random_seed', type=int, default=1)
        parser.add_argument('--yaml_path', type=str, default="./config/detection/bltm_detect/cifar10.yaml", help='config path')
        
        # BLTM 特有参数
        parser.add_argument('--rho', type=float, help='estimated noise rate for thresholding')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/bltm_detect/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path) 
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoints/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        
        print(f"Loading result from {attack_file}/attack_result.pt")
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        if not logger.handlers:
            fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logFormatter)
            logger.addHandler(consoleHandler)
            logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

    def set_devices(self):
        self.device = torch.device(self.args.device)

    def warmup_classifier(self, train_loader):
        """
        Phase 1: Warm-up Classifier
        """
        print(f"===> Initializing resnet.ResNet18 and warming up for {self.args.warmup_epochs} epochs...")
        
        #model = resnet.ResNet34(self.args.num_classes)
        model = preact_resnet.PreActResNet18(self.args.num_classes)
            
        model.to(self.device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.sgd_momentum)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.args.warmup_epochs + 1):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Warmup Epoch {epoch}/{self.args.warmup_epochs}")
            for data, target, *others in pbar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
            avg_loss = total_loss / len(train_loader)
            avg_acc = 100. * correct / total
            logging.info(f'Warmup Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%')

        warmup_save_path = os.path.join(self.args.save_path, 'warmup_classifier.pth')
        torch.save(model.state_dict(), warmup_save_path)
        print(f"Warmup classifier saved to {warmup_save_path}")
        
        return model

    def distill_samples(self, classifier, train_loader):
        """
        Phase 2: Distillation
        """
        print("===> Distilling samples for T-Net training...")
        classifier.eval()
        threshold = (1 + self.args.rho) / 2
        #threshold = 0.01
        print(f"Distillation Threshold: {threshold:.4f}")

        distilled_images = []
        distilled_bayes_labels = [] 
        distilled_noisy_labels = []
        
        # 统计计数器
        total_samples = 0
        distilled_count = 0
        
        total_poison_in_dataset = 0     # 数据集中总共的中毒样本
        selected_poison_count = 0       # 被选入 T-Net 训练集的中毒样本
        selected_clean_count = 0        # 被选入 T-Net 训练集的干净样本

        with torch.no_grad():
            # 这里直接解包 loader 返回的 5 个元素
            # img: [B, C, H, W]
            # target: [B] (这是训练用的标签，如果是中毒样本，这里是攻击目标标签)
            # original_index: [B]
            # poison_indicator: [B] (0=clean, 1=poisoned)
            # original_target: [B] (原始真实标签)
            for batch_idx, (data, target, original_index, poison_indicator, original_target) in enumerate(tqdm(train_loader, desc="Distilling")):
                
                data = data.to(self.device)
                target = target.to(self.device)
                poison_indicator = poison_indicator.to(self.device)
                
                # 统计当前 batch 的中毒总数
                total_poison_in_dataset += poison_indicator.sum().item()

                # 1. Warm-up 模型预测
                outputs = classifier(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probs = F.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                # 2. 生成筛选掩码 (Mask): 只保留模型非常确信的样本
                mask = max_probs > threshold
                
                # 3. 收集数据
                if mask.sum() > 0:
                    distilled_images.append(data[mask].cpu())
                    distilled_bayes_labels.append(preds[mask].cpu())
                    distilled_noisy_labels.append(target[mask].cpu())
                    
                    # --- 统计逻辑 ---
                    # 提取被选中样本的中毒指示器
                    selected_indicator = poison_indicator[mask]
                    
                    p_num = selected_indicator.sum().item() # 选中的中毒数
                    c_num = mask.sum().item() - p_num       # 选中的干净数
                    
                    selected_poison_count += p_num
                    selected_clean_count += c_num
                
                total_samples += data.size(0)
                distilled_count += mask.sum().item()

        if distilled_count == 0:
            raise ValueError(f"No samples passed threshold {threshold}. Try lowering --rho.")

        images = torch.cat(distilled_images)
        bayes_labels = torch.cat(distilled_bayes_labels)
        noisy_labels = torch.cat(distilled_noisy_labels)

        # --- 打印详细统计报告 ---
        print("\n" + "="*50)
        print(f"Distillation Report (rho={self.args.rho}, threshold={threshold:.4f})")
        print("-" * 20)
        print(f"[-] Dataset Total Samples : {total_samples}")
        print(f"[-] Dataset Total Poisons : {int(total_poison_in_dataset)}")
        print("-" * 20)
        print(f"[+] Distilled Set Size    : {distilled_count} (Retention: {distilled_count/total_samples:.2%})")
        print(f"[+] Selected Clean        : {int(selected_clean_count)}")
        print(f"[+] Selected Poisoned     : {int(selected_poison_count)}")
        
        if total_poison_in_dataset > 0:
            recall = selected_poison_count / total_poison_in_dataset
            print(f"[!] Poison Recall         : {recall:.2%} (Poisons captured / Total poisons)")
        
        if distilled_count > 0:
            precision = selected_poison_count / distilled_count
            print(f"[!] Poison Precision      : {precision:.2%} (Poisons / Distilled Set)")
        print("="*50 + "\n")

        return TensorDataset(images, bayes_labels, noisy_labels)

    
    def train(self):
        print("===> Starting BLTM Training Process...")
        self.set_devices()
        fix_random(self.args.random_seed)
        self.set_logger()
        args = self.args
        
        # 1. 加载数据
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train=True)
        bd_dataset = self.result['bd_train'].wrapped_dataset
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        # 2. Warm-up Classifier
        classifier = self.warmup_classifier(train_loader)

        # 3. Distillation
        distilled_dataset = self.distill_samples(classifier, train_loader)
        distilled_loader = DataLoader(
            distilled_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        # 4. 初始化 T-Net
        print("===> Initializing Bayes Label Transition Network (T-Net)...")
        # 这里的 num_classes 需要传入 C*C，因为Linear 层定义是 out_features=num_classes
        t_net_output_dim = args.num_classes * args.num_classes
        
        #T_model = resnet_bayes.ResNet34(t_net_output_dim)
        T_model = preact_resnet_T.PreActResNet18(t_net_output_dim)
        
        
        cls_state = classifier.state_dict()
        t_state = T_model.state_dict()
        
        # 过滤掉不匹配的层 (主要就是 'linear.weight' vs 'bayes_linear.weight')
        pretrained_dict = {k: v for k, v in cls_state.items() if k in t_state and v.size() == t_state[k].size()}
        t_state.update(pretrained_dict)
        T_model.load_state_dict(t_state)
        
        print(f"Initialized T-Net using Warm-up Classifier weights ({len(pretrained_dict)} layers matched).")

        T_model.to(self.device)
        T_model.train()

        # # 5. 训练 T-Net
        optimizer = optim.SGD(T_model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
        loss_function = nn.NLLLoss()

        print(f"===> Start Training T-Net for {args.epochs} epochs...")
        
        patience = 5  # 容忍多少个epoch没有提升
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            total_loss = 0
            pbar = tqdm(enumerate(distilled_loader), total=len(distilled_loader), desc=f"Epoch {epoch}/{args.epochs}")
            
            for batch_idx, (data, bayes_labels, noisy_labels) in pbar:
                data = data.to(self.device)
                bayes_labels = bayes_labels.to(self.device)
                noisy_labels = noisy_labels.to(self.device)
                                                    
                T_pred = T_model(data) 
                bayes_one_hot = F.one_hot(bayes_labels, args.num_classes).float().unsqueeze(1)
                noisy_class_post = torch.bmm(bayes_one_hot, T_pred).squeeze(1)
                log_noisy_class_post = torch.log(noisy_class_post + 1e-12)
                loss = loss_function(log_noisy_class_post, noisy_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(distilled_loader)
            print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

            # 早停判断
            if avg_loss < best_loss - 1e-6: 
                best_loss = avg_loss
                epochs_no_improve = 0
                # 可选：保存最优模型
                torch.save(T_model.state_dict(), os.path.join(self.args.save_path, 'T_model_best.pth'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                    break
        
        final_save_file = os.path.join(self.args.save_path, 'T_model_final.pth')
        torch.save(T_model.state_dict(), final_save_file)
        print(f"Saved T-Net to {final_save_file}")

    def save_all_matrices(self):
        print("===> Generating Transition Matrices for ALL samples...")
        self.set_devices()
        args = self.args
        
        t_net_output_dim = args.num_classes * args.num_classes
        
        #T_model = resnet_bayes.ResNet34(t_net_output_dim)
        T_model = preact_resnet_T.PreActResNet18(t_net_output_dim)

        model_path = os.path.join(args.save_path, 'T_model_final.pth')
        if not os.path.exists(model_path):
            print("Model file not found, please train first.")
            return
        
        T_model.load_state_dict(torch.load(model_path, map_location=self.device))
        T_model.to(self.device)
        T_model.eval()
        
        print("###Loading Warm-up Classifier.....")
        #C_model = resnet.ResNet34(args.num_classes)
        C_model = preact_resnet.PreActResNet18(args.num_classes)
                
        warmup_path = os.path.join(args.save_path, 'warmup_classifier.pth')
        if not os.path.exists(warmup_path):
            print("Warm-up Classifier file not found, please train first.")
            return
        C_model.load_state_dict(torch.load(warmup_path, map_location=self.device))
        C_model.to(self.device)
        C_model.eval()

        train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        bd_dataset = self.result['bd_train'].wrapped_dataset
        poison_indicator = np.array(bd_dataset.poison_indicator)
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        all_T = []
        all_original_targets = [] 
        all_bayes_labels = []

        with torch.no_grad():
            for data, target, original_index, poison_indicator_batch, original_target in tqdm(train_loader, desc="Inference"):
                data = data.to(self.device)
                T_pred = T_model(data)
                
                all_T.append(T_pred.cpu().numpy())
                all_original_targets.append(original_target.cpu().numpy())
                
                # Classifier 推断 (获取 Bayes Label)
                cls_out = C_model(data)
                if isinstance(cls_out, tuple): cls_out = cls_out[0]
                
                # 获取预测类别
                probs = F.softmax(cls_out, dim=1)
                _, bayes_preds = torch.max(probs, dim=1)
                all_bayes_labels.append(bayes_preds.cpu().numpy())
        
        all_T = np.concatenate(all_T, axis=0)
        all_original_targets = np.concatenate(all_original_targets, axis=0)
        all_bayes_labels = np.concatenate(all_bayes_labels, axis=0)
        all_index = np.arange(len(all_T))
        all_is_poison = poison_indicator[:len(all_T)]

        
        # 1. 总转移矩阵 
        total_sum_T = np.sum(all_T, axis=0)
        
        # 2. 行归一化矩阵 (Row Normalized)
        row_sums = np.sum(total_sum_T, axis=1, keepdims=True)
        norm_total_T = total_sum_T / (row_sums + 1e-12)

        # 保存 npz
        npz_path = os.path.join(args.save_path, "all_T_matrix.npz")
        np.savez(npz_path, 
                 all_index=all_index, 
                 all_is_poison=all_is_poison, 
                 all_T=all_T, 
                 all_original_targets=all_original_targets,
                 all_bayes_labels=all_bayes_labels,
                 total_sum_T=total_sum_T,  
                 norm_total_T=norm_total_T)
        print(f"Saved binary matrices to {npz_path}")

        # 保存 txt
        txt_path = os.path.join(args.save_path, "all_T_matrix.txt")
        print(f"Writing text report to {txt_path} ...")
        
        with open(txt_path, "w") as f:
            # 写入每个样本的详情
            for idx, is_p, T, orig_label , bayes_lbl in zip(all_index, all_is_poison, all_T, all_original_targets, all_bayes_labels):
                f.write(f"Index: {idx}\n")
                f.write(f"Is_Poison: {is_p}\n")
                f.write(f"Original_Label: {orig_label}\n")
                f.write(f"Bayes_Label: {bayes_lbl}\n")
                f.write("Transition Matrix:\n")
                for row in T:
                    f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
                f.write("-------------------\n")
            
            f.write("\n=== Summary ===\n")
            
            # 写入总和矩阵
            f.write("1. Total Transition Matrix (Sum of All Matrices):\n")
            for row in total_sum_T:
                f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
            f.write("-------------------\n")

            # 写入行归一化后的总矩阵
            f.write("2. Total Transition Matrix (Row Normalized / Global Average):\n")
            for row in norm_total_T:
                f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
            f.write("-------------------\n")
            
            # 分类统计：中毒样本平均
            if np.sum(all_is_poison) > 0:
                poison_T = all_T[all_is_poison == 1]
                avg_poison_T = np.mean(poison_T, axis=0)
                f.write("3. Average Transition Matrix (Poisoned Samples Only):\n")
                for row in avg_poison_T:
                    f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
                f.write("-------------------\n")
            
            # 分类统计：干净样本平均
            if np.sum(all_is_poison == 0) > 0:
                clean_T = all_T[all_is_poison == 0]
                avg_clean_T = np.mean(clean_T, axis=0)
                f.write("4. Average Transition Matrix (Clean Samples Only):\n")
                for row in avg_clean_T:
                    f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
                    
        print("Done.")
        
    def visualize_target_class_features(self):
            """
            Phase 5: Visualize target class samples using t-SNE
            """
            print("===> Visualizing Target Class Samples with t-SNE...")
            
            # 加载 npz 获取转移矩阵和标签
            npz_path = os.path.join(self.args.save_path, 'all_T_matrix.npz')
            if not os.path.exists(npz_path):
                print("Error: all_T_matrix.npz not found. Please run save_all_matrices first.")
                return
                
            data = np.load(npz_path)
            all_T = data['all_T']
            all_is_poison = data['all_is_poison']
            
            if 'norm_total_T' in data:
                norm_total_T = data['norm_total_T']
            else:
                total_sum_T = np.sum(all_T, axis=0)
                row_sums = np.sum(total_sum_T, axis=1, keepdims=True)
                norm_total_T = total_sum_T / (row_sums + 1e-12)
            
            # 确定目标类
            num_classes = self.args.num_classes
            mask_off_diag = 1 - np.eye(num_classes)
            off_diag_sums = np.sum(norm_total_T * mask_off_diag, axis=0)
            predicted_target_class = np.argmax(off_diag_sums)
            
            print(f"Target Class: {predicted_target_class}")
            
            # 加载 warmup classifier 模型
            print("Loading warmup classifier...")
            C_model = preact_resnet.PreActResNet18(num_classes)
            warmup_path = os.path.join(self.args.save_path, 'warmup_classifier.pth')
            if not os.path.exists(warmup_path):
                print("Error: warmup_classifier.pth not found.")
                return
            C_model.load_state_dict(torch.load(warmup_path, map_location=self.device))
            C_model.to(self.device)
            C_model.eval()
            
            # 构建数据加载器
            train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
            bd_dataset = self.result['bd_train'].wrapped_dataset
            bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
            
            train_loader = DataLoader(
                bd_train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory
            )
            
            # 提取特征和预测
            all_features = []
            all_preds = []
            
            print("Extracting features from warmup classifier...")
            with torch.no_grad():
                for batch_idx, (data, target, *others) in enumerate(tqdm(train_loader, desc="Feature Extraction")):
                    data = data.to(self.device)
                    
                    # 获取倒数第二层特征（avgpool 之后、linear 之前）
                    # 按照 PreActResNet 的结构提取特征
                    x = C_model.conv1(data)
                    x = C_model.layer1(x)
                    x = C_model.layer2(x)
                    x = C_model.layer3(x)
                    x = C_model.layer4(x)
                    x = C_model.avgpool(x)
                    features = x.view(x.size(0), -1)  # 展平为 [B, 512]
                    
                    # 获取预测类别
                    logits = C_model.linear(features)
                    probs = F.softmax(logits, dim=1)
                    _, preds = torch.max(probs, dim=1)
                    
                    all_features.append(features.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
            
            all_features = np.concatenate(all_features, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            
            # 筛选模型预测为目标类的样本
            target_class_mask = all_preds == predicted_target_class
            target_features = all_features[target_class_mask]
            target_labels = all_is_poison[target_class_mask]
            
            num_target_samples = np.sum(target_class_mask)
            num_poison_target = np.sum(target_labels)
            num_clean_target = num_target_samples - num_poison_target
            
            print(f"Samples predicted as target class {predicted_target_class}: {num_target_samples}")
            print(f"  - Clean: {num_clean_target}")
            print(f"  - Poisoned: {num_poison_target}")
            
            if num_target_samples < 2:
                print("Error: Too few target class samples for visualization.")
                return
            
            # 应用 t-SNE
            print("Applying t-SNE (perplexity=30, n_iter=1000)...")
            from sklearn.manifold import TSNE
            perplexity = min(30, max(5, num_target_samples // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            features_2d = tsne.fit_transform(target_features)
            
            # 绘制散点图
            plt.figure(figsize=(12, 8))
            
            clean_mask = target_labels == 0
            poison_mask = target_labels == 1
            
            plt.scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                        c='green', label=f'Clean ({num_clean_target})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            plt.scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                        c='red', label=f'Poisoned ({num_poison_target})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            plt.title(f't-SNE Visualization: Target Class {predicted_target_class}\n(Warmup Classifier Feature Space)',
                    fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE Component 1', fontsize=12)
            plt.ylabel('t-SNE Component 2', fontsize=12)
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join(self.args.save_path, 'tsne_target_class_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   -> [Saved] t-SNE visualization to {plot_path}")
            
            # 计算聚集质量指标
            print("\n[t-SNE Clustering Quality Metrics]")
            if num_clean_target > 1 and num_poison_target > 1:
                clean_center = np.mean(features_2d[clean_mask], axis=0)
                poison_center = np.mean(features_2d[poison_mask], axis=0)
                
                clean_intra_dist = np.mean([np.linalg.norm(features_2d[i] - clean_center)
                                            for i in np.where(clean_mask)[0]])
                poison_intra_dist = np.mean([np.linalg.norm(features_2d[i] - poison_center)
                                            for i in np.where(poison_mask)[0]])
                inter_dist = np.linalg.norm(clean_center - poison_center)
                
                print(f"  - Clean intra-distance: {clean_intra_dist:.4f}")
                print(f"  - Poison intra-distance: {poison_intra_dist:.4f}")
                print(f"  - Inter-class distance: {inter_dist:.4f}")
                avg_intra = (clean_intra_dist + poison_intra_dist) / 2
                if avg_intra > 0:
                    cluster_ratio = inter_dist / avg_intra
                    print(f"  - Cluster Ratio (inter/avg_intra): {cluster_ratio:.4f} (higher is better)")
    
    def visualize_kmeans_clustering(self):
        """
        可视化 K-Means 聚类结果，带 t-SNE 降维
        """
        
        
        print("===> Visualizing K-Means Clustering Results with t-SNE...")
        
        # 1. 加载特征和标签（同 t-SNE 可视化流程）
        npz_path = os.path.join(self.args.save_path, 'all_T_matrix.npz')
        data = np.load(npz_path)
        all_is_poison = data['all_is_poison']
        
        # 2. 提取特征
        num_classes = self.args.num_classes
        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        bd_dataset = self.result['bd_train'].wrapped_dataset
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        train_loader = DataLoader(
            bd_train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )
        
        # 加载 warmup classifier
        C_model = preact_resnet.PreActResNet18(num_classes)
        warmup_path = os.path.join(self.args.save_path, 'warmup_classifier.pth')
        C_model.load_state_dict(torch.load(warmup_path, map_location=self.device))
        C_model.to(self.device)
        C_model.eval()
        
        # 3. 提取目标类特征和预测
        norm_total_T = data.get('norm_total_T', None)
        if norm_total_T is None:
            total_sum_T = np.sum(data['all_T'], axis=0)
            row_sums = np.sum(total_sum_T, axis=1, keepdims=True)
            norm_total_T = total_sum_T / (row_sums + 1e-12)
        
        mask_off_diag = 1 - np.eye(num_classes)
        off_diag_sums = np.sum(norm_total_T * mask_off_diag, axis=0)
        predicted_target_class = np.argmax(off_diag_sums)
        
        all_features = []
        all_preds = []
        
        print("Extracting features...")
        with torch.no_grad():
            for batch_idx, (data_batch, target, *others) in enumerate(tqdm(train_loader, desc="Feature Extraction")):
                data_batch = data_batch.to(self.device)
                
                x = C_model.conv1(data_batch)
                x = C_model.layer1(x)
                x = C_model.layer2(x)
                x = C_model.layer3(x)
                x = C_model.layer4(x)
                x = C_model.avgpool(x)
                features = x.view(x.size(0), -1)
                
                logits = C_model.linear(features)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                
                all_features.append(features.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        # 4. 筛选目标类样本
        target_class_mask = all_preds == predicted_target_class
        target_features = all_features[target_class_mask]
        target_labels = all_is_poison[target_class_mask]
        
        num_target_samples = np.sum(target_class_mask)
        print(f"Target Class: {predicted_target_class}, Samples: {num_target_samples}")
        
        # 5. 在 512 维空间中进行 K-Means 聚类
        print("Running K-Means clustering (k=2)...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(target_features)
        
        # print("Running Gaussian Mixture Model clustering (k=2)...")
        # gmm = GaussianMixture(
        #     n_components=2,
        #     covariance_type='full',  # 关键：允许不同形状
        #     n_init=10,
        #     random_state=42
        # )
        # kmeans_labels = gmm.fit_predict(target_features)
        
        # 6. 进行 t-SNE 降维（用于可视化）
        print("Applying t-SNE...")
        perplexity = min(30, max(5, num_target_samples // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(target_features)
        
        # 7. 绘制对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- 左图：真实标签 ---
        clean_mask = target_labels == 0
        poison_mask = target_labels == 1
        
        axes[0].scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                        c='green', label=f'Clean ({np.sum(clean_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                        c='red', label=f'Poisoned ({np.sum(poison_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_title(f'Ground Truth Labels\n(Clean vs Poisoned)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # --- 右图：K-Means 预测聚类 ---
        cluster_0_mask = kmeans_labels == 0
        cluster_1_mask = kmeans_labels == 1
        
        axes[1].scatter(features_2d[cluster_0_mask, 0], features_2d[cluster_0_mask, 1],
                        c='blue', label=f'Cluster 0 ({np.sum(cluster_0_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1].scatter(features_2d[cluster_1_mask, 0], features_2d[cluster_1_mask, 1],
                        c='orange', label=f'Cluster 1 ({np.sum(cluster_1_mask)})', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1].set_title(f'K-Means Predictions (k=2)\n(Cluster 0 vs Cluster 1)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('t-SNE Component 1')
        axes[1].set_ylabel('t-SNE Component 2')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Target Class {predicted_target_class}: Ground Truth vs K-Means Clustering', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 8. 保存图表
        plot_path = os.path.join(self.args.save_path, 'kmeans_vs_groundtruth.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   -> [Saved] K-Means visualization to {plot_path}")
        
        # 9. 评估聚类效果
        print("\n[K-Means Clustering Evaluation]")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 尝试两种对应关系（聚类 0 vs 1 都可能对应 clean/poison）
        accuracy_1 = accuracy_score(target_labels, kmeans_labels)
        accuracy_2 = accuracy_score(target_labels, 1 - kmeans_labels)
        
        best_accuracy = max(accuracy_1, accuracy_2)
        best_kmeans_labels = kmeans_labels if accuracy_1 >= accuracy_2 else 1 - kmeans_labels
        
        print(f"  - Accuracy: {best_accuracy:.4f}")
        print(f"  - Precision: {precision_score(target_labels, best_kmeans_labels, zero_division=0):.4f}")
        print(f"  - Recall: {recall_score(target_labels, best_kmeans_labels, zero_division=0):.4f}")
        print(f"  - F1-Score: {f1_score(target_labels, best_kmeans_labels, zero_division=0):.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BLTM T-Net on Backdoored Data')  
    BLTMTrainer.add_arguments(parser)
    args = parser.parse_args()

    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'

    trainer = BLTMTrainer(args)
    
    t_model_path = os.path.join(trainer.args.save_path, 'T_model_final.pth')
    
    if os.path.exists(t_model_path):
        print("Detected trained T-Net model. Skipping training phase.")
        trainer.save_all_matrices()
    else:
        trainer.train()
        trainer.save_all_matrices()
        
    trainer.visualize_target_class_features()
    trainer.visualize_kmeans_clustering()