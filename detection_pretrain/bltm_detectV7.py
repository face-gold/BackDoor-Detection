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
    
    def visualize_kmeans_clustering_v2_with_p1_union(self):
        """
        增强版本：结合P1筛选和双特征K-Means
        - P1: 基于转移矩阵非对角线元素和的可疑样本
        - U: P1 ∪ 预测为目标类的样本集合
        - 比较两种聚类方式：
        1. 标准K-Means（512维特征空间）
        2. 自定义K-Means（512维特征 + 转移矩阵特征）
        """
        
        print("="*60)
        print("Enhanced K-Means: P1 Union + Dual-Feature Clustering")
        print("="*60)
        
        # ==== Step 1: 加载数据 ====
        npz_path = os.path.join(self.args.save_path, 'all_T_matrix.npz')
        if not os.path.exists(npz_path):
            print("Error: all_T_matrix.npz not found.")
            return
            
        data = np.load(npz_path)
        all_T = data['all_T']
        all_is_poison = data['all_is_poison']
        
        norm_total_T = data.get('norm_total_T', None)
        if norm_total_T is None:
            total_sum_T = np.sum(all_T, axis=0)
            row_sums = np.sum(total_sum_T, axis=1, keepdims=True)
            norm_total_T = total_sum_T / (row_sums + 1e-12)
        
        num_classes = self.args.num_classes
        num_samples = all_T.shape[0]
        
        # ==== Step 2: 确定目标类 ====
        mask_off_diag = 1 - np.eye(num_classes)
        global_off_diag_sums = np.sum(norm_total_T * mask_off_diag, axis=0)
        predicted_target_class = np.argmax(global_off_diag_sums)
        print(f"\n[Step 1] Target Class: {predicted_target_class}")
        
        # ==== Step 3: 筛选P1集合（基于转移矩阵） ====
        print("\n[Step 2] Filtering Suspicious Set (P1)...")
        p1_indices = []
        p1_off_diag_scores = []  # 每个样本在目标类上的非对角线元素和
        
        for i in range(num_samples):
            # 计算该样本的非对角线元素和
            sample_off_diag_sums = np.sum(all_T[i] * mask_off_diag, axis=0)
            
            # 找最大列
            max_col_idx = np.argmax(sample_off_diag_sums)
            
            # 如果最大列是目标类，则加入P1
            if max_col_idx == predicted_target_class:
                p1_indices.append(i)
                p1_off_diag_scores.append(sample_off_diag_sums[predicted_target_class])
        
        p1_indices = np.array(p1_indices)
        p1_off_diag_scores = np.array(p1_off_diag_scores)
        print(f"  P1 Set Size: {len(p1_indices)}")
        print(f"  P1 Samples: {len(p1_indices)} / {num_samples} ({100*len(p1_indices)/num_samples:.2f}%)")
        
        # ==== Step 4: 提取目标类预测的样本特征 ====
        print("\n[Step 3] Extracting Target Class Features...")
        
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
        
        C_model = preact_resnet.PreActResNet18(num_classes)
        warmup_path = os.path.join(self.args.save_path, 'warmup_classifier.pth')
        C_model.load_state_dict(torch.load(warmup_path, map_location=self.device))
        C_model.to(self.device)
        C_model.eval()
        
        # 提取所有样本的特征和预测
        all_features = []
        all_preds = []
        
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
        
        all_features = np.concatenate(all_features, axis=0)  # [N, 512]
        all_preds = np.concatenate(all_preds, axis=0)        # [N]
        
        # # 使用PCA降维
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=100)  # 保留95%方差
        # u_features_pca = pca.fit_transform(all_features)  # [N, D_pca]
        
        # print(f"  Original Feature Dim: {all_features.shape[1]}, PCA Reduced Dim: {u_features_pca.shape[1]}")
        # all_features = u_features_pca  # 替换为PCA降维后的特征
        
        # ==== Step 5: 构造并集U = P1 ∪ 预测为目标类 ====
        print("\n[Step 4] Computing Union Set U...")
        
        target_class_mask = all_preds == predicted_target_class
        target_class_indices = np.where(target_class_mask)[0]
        
        # 并集：combine P1 and target class predictions
        union_indices = np.union1d(p1_indices, target_class_indices)
        
        #union_indices = p1_indices  # 直接用P1
        #union_indices = target_class_indices  # 直接用预测为目标类的样本
        # 提取U中的样本特征和标签
        u_features = all_features[union_indices]              # [U_size, 512]
        u_labels = all_is_poison[union_indices]               # [U_size]
        
        print(f"    - Clean: {np.sum(u_labels == 0)}")
        print(f"    - Poisoned: {np.sum(u_labels == 1)}")
        
        print(f"  Predicted Target Class: {np.sum(target_class_mask)} samples")
        print(f"  P1 Set: {len(p1_indices)} samples")
        print(f"  Union U: {len(union_indices)} samples")
        
                
        # 提取U中每个样本的转移矩阵特征（非对角线元素和）
        u_matrix_features = []
        for idx in union_indices:
            sample_off_diag_sums = np.sum(all_T[idx] * mask_off_diag, axis=0)
            u_matrix_features.append(sample_off_diag_sums[predicted_target_class])
        
        u_matrix_features = np.array(u_matrix_features)       # [U_size]
        
        # ==== Step 6: 方法1 - 标准K-Means GMM====
        print("\n[Step 5] Method 1: Standard K-Means (512-dim feature space)...")
        
        # kmeans_std = KMeans(n_clusters=2, random_state=42, n_init=10)
        # kmeans_std_labels = kmeans_std.fit_predict(u_features)
        
        gmm_std = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full')
        kmeans_std_labels = gmm_std.fit_predict(u_features)
        
        # ==== Step 7: 方法2 - 自定义双特征K-Means ====
        print("\n[Step 6] Method 2: Custom Dual-Feature K-Means...")
        
        # 特征规范化
        u_features_norm = (u_features - u_features.mean(axis=0)) / (u_features.std(axis=0) + 1e-10)
        u_matrix_features_norm = (u_matrix_features - u_matrix_features.mean()) / (u_matrix_features.std() + 1e-10)
        
        # 组合特征：512维 + 1维 = 513维
        combined_features = np.concatenate([
            u_features_norm,
            u_matrix_features_norm.reshape(-1, 1)
        ], axis=1)  # [U_size, 513]
        
        print(f"  Combined Feature Shape: {combined_features.shape}")
        
        kmeans_dual = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_dual_labels = kmeans_dual.fit_predict(combined_features)
        
        # ==== Step 8: 自定义实现K-Means（展示细节） ====
        print("\n[Step 7] Custom K-Means Implementation (detailed)...")
        
        custom_kmeans_labels = self._custom_kmeans_clustering(
            combined_features,
            n_clusters=2,
            max_iter=100,
            random_seed=42
        )
        
        # === 仅通过转移矩阵特征进行聚类 ====
        print("\n[Step 7.1] Clustering Based on Transition Matrix Feature Only...")
        kmeans_matrix_only = KMeans(n_clusters=2, random_state=42, n_init=10)
        gmm_matrix_only = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full')
        kmeans_matrix_only_labels = gmm_matrix_only.fit_predict(u_matrix_features.reshape(-1, 1))
        
        # ==== Step 9: t-SNE 可视化用于对比 ====
        print("\n[Step 8] Applying t-SNE for visualization...")
        
        perplexity = min(30, max(5, len(union_indices) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(u_features)  # 基于原始512维特征进行降维
        
        # ==== Step 10: 绘制三行对比图 ====
        print("\n[Step 9] Generating comparison visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # ===== 第一行：真实标签 vs 标准K-Means vs 双特征K-Means =====
        
        # (0,0) 真实标签
        clean_mask = u_labels == 0
        poison_mask = u_labels == 1
        
        axes[0, 0].scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                        c='green', label=f'Clean ({np.sum(clean_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 0].scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                        c='red', label=f'Poisoned ({np.sum(poison_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 0].set_title('Ground Truth Labels', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # (0,1) 标准K-Means
        cluster_0_mask = kmeans_std_labels == 0
        cluster_1_mask = kmeans_std_labels == 1
        
        axes[0, 1].scatter(features_2d[cluster_0_mask, 0], features_2d[cluster_0_mask, 1],
                        c='blue', label=f'Cluster 0 ({np.sum(cluster_0_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 1].scatter(features_2d[cluster_1_mask, 0], features_2d[cluster_1_mask, 1],
                        c='orange', label=f'Cluster 1 ({np.sum(cluster_1_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 1].set_title('Standard K-Means (512-dim)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # (0,2) 双特征K-Means
        cluster_0_mask = kmeans_dual_labels == 0
        cluster_1_mask = kmeans_dual_labels == 1
        
        axes[0, 2].scatter(features_2d[cluster_0_mask, 0], features_2d[cluster_0_mask, 1],
                        c='cyan', label=f'Cluster 0 ({np.sum(cluster_0_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 2].scatter(features_2d[cluster_1_mask, 0], features_2d[cluster_1_mask, 1],
                        c='magenta', label=f'Cluster 1 ({np.sum(cluster_1_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[0, 2].set_title('Dual-Feature K-Means (512+1 dim)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('t-SNE 1')
        axes[0, 2].set_ylabel('t-SNE 2')
        axes[0, 2].legend(fontsize=9)
        axes[0, 2].grid(True, alpha=0.3)
        
        # ===== 第二行：覆盖对比 =====
        
        # (1,0) Ground Truth
        axes[1, 0].scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                        c='green', label=f'Clean ({np.sum(clean_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 0].scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                        c='red', label=f'Poisoned ({np.sum(poison_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 0].set_title('Ground Truth (Reference)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # (1,1) Custom K-Means
        cluster_0_mask = custom_kmeans_labels == 0
        cluster_1_mask = custom_kmeans_labels == 1
        
        axes[1, 1].scatter(features_2d[cluster_0_mask, 0], features_2d[cluster_0_mask, 1],
                        c='purple', label=f'Cluster 0 ({np.sum(cluster_0_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 1].scatter(features_2d[cluster_1_mask, 0], features_2d[cluster_1_mask, 1],
                        c='yellow', label=f'Cluster 1 ({np.sum(cluster_1_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 1].set_title('Custom K-Means Implementation', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # # (1,2) P1分布
        # p1_mask = np.isin(union_indices, p1_indices)
        # not_p1_mask = ~p1_mask
        
        # axes[1, 2].scatter(features_2d[not_p1_mask, 0], features_2d[not_p1_mask, 1],
        #                 c='lightblue', label=f'Not P1 ({np.sum(not_p1_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        # axes[1, 2].scatter(features_2d[p1_mask, 0], features_2d[p1_mask, 1],
        #                 c='darkred', label=f'P1 Suspicious ({np.sum(p1_mask)})', alpha=0.8, s=30, edgecolors='black', linewidth=0.3)
        # axes[1, 2].set_title('P1 Suspicious Samples Distribution', fontsize=12, fontweight='bold')
        # axes[1, 2].set_xlabel('t-SNE 1')
        # axes[1, 2].set_ylabel('t-SNE 2')
        # axes[1, 2].legend(fontsize=9)
        # axes[1, 2].grid(True, alpha=0.3)
        
        # (1,2) 仅转移矩阵特征聚类
        cluster_0_mask = kmeans_matrix_only_labels == 0
        cluster_1_mask = kmeans_matrix_only_labels == 1

        axes[1, 2].scatter(features_2d[cluster_0_mask, 0], features_2d[cluster_0_mask, 1],
                        c='cyan', label=f'Cluster 0 ({np.sum(cluster_0_mask)})', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 2].scatter(features_2d[cluster_1_mask, 0], features_2d[cluster_1_mask, 1],
                        c='orange', label=f'Cluster 1 ({np.sum(cluster_1_mask)})', alpha=0.8, s=30, edgecolors='black', linewidth=0.3)
        axes[1, 2].set_title('Matrix-Feature-Only K-Means Clustering', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('t-SNE 1')
        axes[1, 2].set_ylabel('t-SNE 2')
        axes[1, 2].legend(fontsize=9)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Enhanced Clustering on Union U: P1 ∪ Target Class\nTarget Class {predicted_target_class}, Union Size: {len(union_indices)}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = os.path.join(self.args.save_path, 'kmeans_p1_union_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   -> [Saved] Comparison visualization to {plot_path}")
        
        # ==== Step 11: 评估三种方式 ====
        print("\n" + "="*60)
        print("[EVALUATION RESULTS]")
        print("="*60)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        def align_and_evaluate_with_matrix_feature(pred_labels, true_labels, u_matrix_features, method_name):
            """
            使用目标列非对角线和特征直接判定聚类对应关系
            
            Args:
                pred_labels: K-Means预测标签 [N]
                true_labels: 真实标签 (0=clean, 1=poison) [N]
                u_matrix_features: 目标列上的非对角线元素和 [N]
                method_name: 方法名称
            """
            
            # Step 1: 计算两个簇的平均矩阵特征
            cluster_0_avg_matrix_feature = u_matrix_features[pred_labels == 0].mean()
            cluster_1_avg_matrix_feature = u_matrix_features[pred_labels == 1].mean()
            
            # Step 2: 非对角线和更高的簇 → 中毒样本
            if cluster_0_avg_matrix_feature > cluster_1_avg_matrix_feature:
                # Cluster 0 的非对角线和更高 → Cluster 0 对应中毒样本
                aligned_labels = 1 - pred_labels
                poison_cluster = 0
            else:
                # Cluster 1 的非对角线和更高 → 需要翻转标签
                aligned_labels = pred_labels
                poison_cluster = 1
            
    
            # # ===== Step 1: 计算每个样本属于两个簇的概率 =====
            # c0_features = u_matrix_features[pred_labels == 0]
            # c1_features = u_matrix_features[pred_labels == 1]
            
            # c0_mean = c0_features.mean()
            # c1_mean = c1_features.mean()
            # c0_std = c0_features.std() + 1e-6
            # c1_std = c1_features.std() + 1e-6
            
            # # 高斯似然
            # c0_likelihood = np.exp(-0.5 * ((u_matrix_features - c0_mean) / c0_std) ** 2) / c0_std
            # c1_likelihood = np.exp(-0.5 * ((u_matrix_features - c1_mean) / c1_std) ** 2) / c1_std
            
            # # 计算后验概率
            # c0_prob = c0_likelihood / (c0_likelihood + c1_likelihood)  # P(poison | feature, pred=0)
            # c1_prob = c1_likelihood / (c0_likelihood + c1_likelihood)  # P(poison | feature, pred=1)
            
            # # ===== Step 2: 样本级别的对齐概率 =====
            # # 对于pred_labels==0的样本：
            # #   - 如果c0_prob高，说明这个簇确实是poison
            # #   - 否则不是poison
            
            # align_prob = np.where(pred_labels == 0, c0_prob, c1_prob)
            
            # # 硬阈值：align_prob > 0.5 表示这个样本的簇是poison
            # poison_cluster = 0 if c0_mean > c1_mean else 1
            # aligned_labels_soft = align_prob.copy()
            # aligned_labels = (align_prob > 0.5).astype(int)
            
            # # ===== Step 3: 信心分数 =====
            # confidence = np.abs(align_prob - 0.5) * 2  # 0~1，1表示非常确定
            
            # print(f"\n{method_name}:")
            # print(f"  Cluster 0: mean={c0_mean:.4f}, std={c0_std:.4f}")
            # print(f"  Cluster 1: mean={c1_mean:.4f}, std={c1_std:.4f}")
            # print(f"  Identified Poison Cluster: {poison_cluster}")
            # print(f"  Average confidence: {confidence.mean():.4f}")
            # print(f"  High confidence samples (>0.9): {np.sum(confidence > 0.9)}")
            # print(f"  Low confidence samples (<0.6): {np.sum(confidence < 0.6)}")
            
            # Step 3: 计算指标
            accuracy = accuracy_score(true_labels, aligned_labels)
            precision = precision_score(true_labels, aligned_labels, zero_division=0)
            recall = recall_score(true_labels, aligned_labels, zero_division=0)
            f1 = f1_score(true_labels, aligned_labels, zero_division=0)
            
            # 计算混淆矩阵
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(true_labels, aligned_labels)
            report = classification_report(true_labels, aligned_labels, target_names=['Clean', 'Poisoned'], zero_division=0)
            tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
            fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
            
            # Step 4: 打印详细结果
            print(f"\n{method_name}:")
            print(f"  Cluster 0 avg matrix feature (target col non-diag): {cluster_0_avg_matrix_feature:.4f}")
            print(f"  Cluster 1 avg matrix feature (target col non-diag): {cluster_1_avg_matrix_feature:.4f}")
            print(f"  → Cluster {poison_cluster} identified as Poisoned (higher non-diag sum)")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")
            print(f"  Classification Report:\n{report}")
            print(f"  True Positive Rate (Recall): {tpr:.4f}")
            print(f"  False Positive Rate: {fpr:.4f}")
            
            
            return accuracy, precision, recall, f1, aligned_labels
        
        def align_and_evaluate_by_intra_distance(features, pred_labels, true_labels, method_name):
            """
            使用簇内距离来判定哪个簇是中毒哪个簇是干净
            
            假设：
            - 簇内距离小 → Clean类（样本紧凑）
            - 簇内距离大 → Poison类（样本分散）
            
            Args:
                features: 特征 [N, D]
                pred_labels: K-Means预测标签 [N]
                true_labels: 真实标签 (0=clean, 1=poison) [N]
                method_name: 方法名称
            """
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            
            # ===== Step 1: 计算两个簇的簇内距离 =====
            intra_distances = {}
            cluster_centers = {}
            
            for label in [0, 1]:
                cluster_points = features[pred_labels == label]
                center = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                
                intra_distances[label] = {
                    'mean': distances.mean(),
                    'std': distances.std(),
                    'max': distances.max(),
                    'min': distances.min()
                }
                cluster_centers[label] = center
            
            # ===== Step 2: 判定簇映射关系 =====
            # 簇内距离小的 → Clean，距离大的 → Poison
            cluster_0_intra_dist = intra_distances[0]['mean']
            cluster_1_intra_dist = intra_distances[1]['mean']
            
            if cluster_0_intra_dist < cluster_1_intra_dist:
                # Cluster 0 更紧凑 → Cluster 0 对应 Clean
                aligned_labels = pred_labels  # 0 stays 0 (clean), 1 stays 1 (poison)
                clean_cluster = 0
                poison_cluster = 1
            else:
                # Cluster 1 更紧凑 → Cluster 1 对应 Clean
                aligned_labels = 1 - pred_labels  # 0→1 (poison), 1→0 (clean)
                clean_cluster = 1
                poison_cluster = 0
            
            # ===== Step 3: 计算性能指标 =====
            accuracy = accuracy_score(true_labels, aligned_labels)
            precision = precision_score(true_labels, aligned_labels, zero_division=0)
            recall = recall_score(true_labels, aligned_labels, zero_division=0)
            f1 = f1_score(true_labels, aligned_labels, zero_division=0)
            
            cm = confusion_matrix(true_labels, aligned_labels)
            report = classification_report(true_labels, aligned_labels, 
                                        target_names=['Clean', 'Poisoned'], zero_division=0)
            tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
            fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
            
            # ===== Step 4: 打印详细结果 =====
            print(f"\n{method_name}:")
            print(f"  [Intra-Cluster Distance Analysis]")
            print(f"  Cluster 0 mean intra-distance: {cluster_0_intra_dist:.4f}")
            print(f"  Cluster 1 mean intra-distance: {cluster_1_intra_dist:.4f}")
            print(f"  → Cluster {clean_cluster} identified as Clean (smaller intra-distance)")
            print(f"  → Cluster {poison_cluster} identified as Poison (larger intra-distance)")
            
            print(f"\n  [Performance Metrics]")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  True Positive Rate (Recall): {tpr:.4f}")
            print(f"  False Positive Rate: {fpr:.4f}")
            
            print(f"\n  [Confusion Matrix]")
            print(f"  {cm}")
            print(f"\n  [Classification Report]")
            print(f"  {report}")
            
            # ===== Step 5: 详细的簇统计 =====
            print(f"\n  [Detailed Cluster Statistics]")
            for label in [0, 1]:
                print(f"  Cluster {label}:")
                print(f"    - Size: {np.sum(pred_labels == label)}")
                print(f"    - Mean intra-distance: {intra_distances[label]['mean']:.4f}")
                print(f"    - Std intra-distance: {intra_distances[label]['std']:.4f}")
                print(f"    - Max intra-distance: {intra_distances[label]['max']:.4f}")
                print(f"    - Min intra-distance: {intra_distances[label]['min']:.4f}")
            
            return accuracy, precision, recall, f1, aligned_labels
        
        #使用转移矩阵特征对齐并评估
        std_acc, std_prec, std_rec, std_f1, std_aligned = align_and_evaluate_with_matrix_feature(
            kmeans_std_labels, u_labels, u_matrix_features, 
            "1. Standard K-Means (512-dim)"
        )

        dual_acc, dual_prec, dual_rec, dual_f1, dual_aligned = align_and_evaluate_with_matrix_feature(
            kmeans_dual_labels, u_labels, u_matrix_features,
            "2. Dual-Feature K-Means (512+1 dim)"
        )

        custom_acc, custom_prec, custom_rec, custom_f1, custom_aligned = align_and_evaluate_with_matrix_feature(
            custom_kmeans_labels, u_labels, u_matrix_features,
            "3. Custom K-Means Implementation"
        )
        
        matrix_acc, matrix_prec, matrix_rec, matrix_f1, matrix_aligned = align_and_evaluate_with_matrix_feature(
            kmeans_matrix_only_labels, u_labels, u_matrix_features,
            "4. Matrix-Feature-Only K-Means"
        )
        
        # 使用簇内距离对齐并评估
        # std_acc, std_prec, std_rec, std_f1, std_aligned = align_and_evaluate_by_intra_distance(
        #     u_features, kmeans_std_labels, u_labels,
        #     "1. Standard K-Means (512-dim)"
        # )

        # dual_acc, dual_prec, dual_rec, dual_f1, dual_aligned = align_and_evaluate_by_intra_distance(
        #     u_features, kmeans_dual_labels, u_labels,
        #     "2. Dual-Feature K-Means (512+1 dim)"
        # )

        # custom_acc, custom_prec, custom_rec, custom_f1, custom_aligned = align_and_evaluate_by_intra_distance(
        #     u_features, custom_kmeans_labels, u_labels,
        #     "3. Custom K-Means Implementation"
        # )
        
        # ==== 对比表 ====
        print("\n" + "="*60)
        print("[COMPARISON TABLE]")
        print("="*60)
        print(f"{'Method':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 83)
        print(f"{'Standard K-Means':<35} {std_acc:<12.4f} {std_prec:<12.4f} {std_rec:<12.4f} {std_f1:<12.4f}")
        print(f"{'Dual-Feature K-Means':<35} {dual_acc:<12.4f} {dual_prec:<12.4f} {dual_rec:<12.4f} {dual_f1:<12.4f}")
        print(f"{'Custom K-Means':<35} {custom_acc:<12.4f} {custom_prec:<12.4f} {custom_rec:<12.4f} {custom_f1:<12.4f}")
        print("="*60)
        
        # 计算改进
        print("\n[IMPROVEMENT over Standard K-Means]")
        print("-" * 60)
        print(f"Dual-Feature K-Means: Recall +{(dual_rec - std_rec)*100:.2f}pp, "
            f"F1 +{(dual_f1 - std_f1)*100:.2f}pp")
        print(f"Custom K-Means:       Recall +{(custom_rec - std_rec)*100:.2f}pp, "
            f"F1 +{(custom_f1 - std_f1)*100:.2f}pp")
        print("="*60 + "\n")


    def _custom_kmeans_clustering(self, X, n_clusters=2, max_iter=100, random_seed=42, verbose=True):
        """
        自定义K-Means实现，用于展示算法细节
        
        Args:
            X: [N, D] 输入特征
            n_clusters: 簇数
            max_iter: 最大迭代次数
            random_seed: 随机种子
            verbose: 是否打印详细信息
        
        Returns:
            labels: [N] 聚类标签
        """
        np.random.seed(random_seed)
        N, D = X.shape
        
        # 1. 随机初始化簇中心
        initial_indices = np.random.choice(N, n_clusters, replace=False)
        centers = X[initial_indices].copy()  # [k, D]
        
        if verbose:
            print(f"  Initial centers selected from random samples")
            print(f"  Feature dimension: {D}")
            print(f"  Number of samples: {N}")
        
        labels = np.zeros(N, dtype=int)
        
        for iteration in range(max_iter):
            # 2. 分配步骤：计算每个样本到各簇中心的距离
            distances = np.zeros((N, n_clusters))
            for k in range(n_clusters):
                # 欧氏距离
                distances[:, k] = np.linalg.norm(X - centers[k], axis=1)
            
            # 分配到最近的簇
            new_labels = np.argmin(distances, axis=1)
            
            # 3. 更新步骤：重新计算簇中心
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                cluster_points = X[new_labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
            
            # 4. 收敛判断
            center_shift = np.linalg.norm(new_centers - centers)
            
            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"    Iteration {iteration}: center_shift={center_shift:.6f}")
            
            if center_shift < 1e-6:
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                labels = new_labels
                break
            
            centers = new_centers
            labels = new_labels
        
        return labels
    

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
    trainer.visualize_kmeans_clustering_v2_with_p1_union()