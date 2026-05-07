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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
import cv2
from PIL import Image
import torchvision.transforms as tv_transforms


sys.path.append('../')
sys.path.append(os.getcwd())

from utils.aggregate_block.fix_random import fix_random
from defense.base import defense
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_transform
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

from bltm_detect import preact_resnet
from bltm_detect import preact_resnet_T


class GaussianBlurTransform:
    """高斯模糊处理"""
    def __init__(self, kernel_size=7, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy().transpose(1, 2, 0)
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(img)
        
        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
        return tv_transforms.ToTensor()(blurred)

class UnsharpMaskTransform:
    """反锐化遮罩 - 保留更多细节（最佳效果）"""
    def __init__(self, kernel_size=5, sigma=1.0, strength=1.2):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.strength = strength
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy().transpose(1, 2, 0)
            if img_np.max() <= 1:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # 创建高斯模糊版本
        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
        
        # 计算 High Pass: 原图 - 模糊
        high_pass = cv2.subtract(img_np, blurred)
        
        # 锐化 = 原图 + strength * high_pass
        sharpened = cv2.addWeighted(img_np, 1.0, high_pass, self.strength, 0)
        
        return tv_transforms.ToTensor()(np.clip(sharpened, 0, 255).astype(np.uint8))

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
        # bd_dataset = self.result['bd_train'].wrapped_dataset
        # bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        # train_tran_base = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train=True)
        # gaussian_blur = GaussianBlurTransform(kernel_size=5, sigma=1.0)
        # train_tran = tv_transforms.Compose([train_tran_base, gaussian_blur])
        # sharpening = UnsharpMaskTransform(kernel_size=5, sigma=1.0, strength=1.2)
        # train_tran = tv_transforms.Compose([train_tran_base, sharpening])
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
        #optimizer = optim.Adam(T_model.parameters(), lr=args.lr)
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
        
        T_model = preact_resnet_T.PreActResNet18(t_net_output_dim)

        model_path = os.path.join(args.save_path, 'T_model_final.pth')
        if not os.path.exists(model_path):
            print("Model file not found, please train first.")
            return
        
        T_model.load_state_dict(torch.load(model_path, map_location=self.device))
        T_model.to(self.device)
        T_model.eval()
        
        print("###Loading Warm-up Classifier.....")
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
        
        # train_tran_base = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        # gaussian_blur = GaussianBlurTransform(kernel_size=5, sigma=1.0)
        # train_tran = tv_transforms.Compose([train_tran_base, gaussian_blur])
        # sharpening = UnsharpMaskTransform(kernel_size=5, sigma=1.0, strength=1.2)
        # train_tran = tv_transforms.Compose([train_tran_base, sharpening])
        # bd_dataset = self.result['bd_train'].wrapped_dataset
        # poison_indicator = np.array(bd_dataset.poison_indicator)
        # bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        all_T = []
        all_original_targets = [] # 原始真实标签
        all_bayes_labels = []
        
        all_train_targets = [] #训练时的标签

        with torch.no_grad():
            for data, target, original_index, poison_indicator_batch, original_target in tqdm(train_loader, desc="Inference"):
                data = data.to(self.device)
                T_pred = T_model(data)
                
                all_T.append(T_pred.cpu().numpy())
                all_original_targets.append(original_target.cpu().numpy())
                all_train_targets.append(target.cpu().numpy())
                
                # Classifier 推断 (获取 Bayes Label)
                cls_out = C_model(data)
                if isinstance(cls_out, tuple): cls_out = cls_out[0]
                
                # 获取预测类别
                probs = F.softmax(cls_out, dim=1)
                _, bayes_preds = torch.max(probs, dim=1)
                all_bayes_labels.append(bayes_preds.cpu().numpy())
        
        all_T = np.concatenate(all_T, axis=0)
        all_original_targets = np.concatenate(all_original_targets, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)
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
                 all_train_targets=all_train_targets,
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
        
   
    def setup_logger(self,save_path, name='BLTM'):
        """设置日志记录器"""
        log_path = os.path.join(save_path, 'kmeans_p1_s1_intersection.log')
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        
        # 日志格式
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def visualize_kmeans_with_p1_s1_intersection(self):
        """
        新方法：结合P1和真实目标类标签的交集进行聚类
        - P1: 基于转移矩阵非对角线元素和的可疑样本
        - S1: 真实标签为目标类的样本集合
        - U: P1 ∩ S1 (交集)
        - 对U进行多种聚类方式对比分析
        """
        logger = self.setup_logger(self.args.save_path)
        
        logger.info("="*60)
        logger.info("Enhanced K-Means: P1 Intersection (P1 ∩ S1)")
        logger.info("="*60)
        
        # ==== Step 1: 加载数据 ====
        npz_path = os.path.join(self.args.save_path, 'all_T_matrix.npz')
        if not os.path.exists(npz_path):
            logger.info("Error: all_T_matrix.npz not found.")
            return
            
        data = np.load(npz_path)
        all_T = data['all_T']
        all_is_poison = data['all_is_poison']
        all_train_targets = data['all_train_targets']
        
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
        logger.info(f"\n[Step 1] Detected Target Class: {predicted_target_class}")
        logger.info(f"  Off-diagonal sum in target class: {global_off_diag_sums[predicted_target_class]:.4f}")
        
        # ==== Step 3: 筛选P1集合（基于转移矩阵异常） ====
        logger.info("\n[Step 2] Filtering Suspicious Set (P1) - Based on T-matrix...")
        p1_indices = []
        
        for i in range(num_samples):
            sample_off_diag_sums = np.sum(all_T[i] * mask_off_diag, axis=0)
            max_col_idx = np.argmax(sample_off_diag_sums)
            
            if max_col_idx == predicted_target_class:
                p1_indices.append(i)
        
        p1_indices = np.array(p1_indices)
        logger.info(f"  P1 Size: {len(p1_indices)} / {num_samples} ({100*len(p1_indices)/num_samples:.2f}%)")
        logger.info(f"  P1 Clean: {np.sum(all_is_poison[p1_indices] == 0)}")
        logger.info(f"  P1 Poison: {np.sum(all_is_poison[p1_indices] == 1)}")
        
        # ==== Step 4: 筛选S1集合（真实标签为目标类） ====
        logger.info("\n[Step 3] Filtering Target Class Set (S1) - Based on training label...")
        s1_mask = all_train_targets == predicted_target_class
        s1_indices = np.where(s1_mask)[0]
        
        logger.info(f"  S1 Size: {len(s1_indices)} / {num_samples} ({100*len(s1_indices)/num_samples:.2f}%)")
        logger.info(f"  S1 Clean: {np.sum(all_is_poison[s1_indices] == 0)}")
        logger.info(f"  S1 Poison: {np.sum(all_is_poison[s1_indices] == 1)}")
        
        # ==== Step 5: 计算交集U = P1 ∩ S1 ====
        logger.info("\n[Step 4] Computing Intersection U = P1 ∩ S1...")
        intersection_indices = np.intersect1d(p1_indices, s1_indices)
        intersection_indices = s1_indices # 仅考虑真实标签为目标类的样本进行分析
        
        logger.info(f"  Intersection Size: {len(intersection_indices)} samples")
        
        if len(intersection_indices) == 0:
            logger.info("\n[WARNING] Intersection is empty!")
            logger.info("  Falling back to Union (P1 ∪ S1)...")
            union_indices = np.union1d(p1_indices, s1_indices)
            u_indices = union_indices
            use_intersection = False
            logger.info(f"  Union Size: {len(u_indices)} samples")
        else:
            u_indices = intersection_indices
            use_intersection = True
        
        # ==== Step 6: 提取特征 ====
        logger.info(f"\n[Step 5] Extracting Features from {len(u_indices)} samples...")
        
        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        # train_tran_base = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        # gaussian_blur = GaussianBlurTransform(kernel_size=5, sigma=1.0)
        # train_tran = tv_transforms.Compose([train_tran_base, gaussian_blur])
        # sharpening = UnsharpMaskTransform(kernel_size=5, sigma=1.0, strength=1.2)
        # train_tran = tv_transforms.Compose([train_tran_base, sharpening])
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
        
        # 提取所有样本的特征
        all_features = []
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
                
                all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)  # [N, 512]
        
        # 提取U中的特征和标签
        u_features = all_features[u_indices]              # [U_size, 512]
        u_labels = all_is_poison[u_indices]               # [U_size]
        u_T = all_T[u_indices]                            # [U_size, C, C]
        
        logger.info(f"  U Clean Samples: {np.sum(u_labels == 0)}")
        logger.info(f"  U Poison Samples: {np.sum(u_labels == 1)}")
        logger.info(f"  U Poison Ratio: {np.sum(u_labels == 1) / len(u_labels):.2%}")
        
        # ==== Step 7: 提取转移矩阵特征 ====
        logger.info(f"\n[Step 6] Computing Transition Matrix Features...")
        u_matrix_features = []
        for idx in u_indices:
            sample_off_diag_sums = np.sum(all_T[idx] * mask_off_diag, axis=0)
            u_matrix_features.append(sample_off_diag_sums)
        
        u_matrix_features = np.array(u_matrix_features)  # [U_size, num_classes]
        logger.info(f"  Matrix Features Shape: {u_matrix_features.shape}")
        
        # ==== Step 8: 进行聚类 ====
        logger.info(f"\n[Step 7] Performing Clustering Analysis...")
        
        # 方法1: 标准特征空间聚类
        logger.info("  Method 1: Standard Feature Space (512-dim)...")
        gmm_std = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full')
        labels_std = gmm_std.fit_predict(u_features)
        
        # 方法2: 结合转移矩阵特征
        logger.info("  Method 2: Dual-Feature Space (512 + 10 dims)...")
        u_features_norm = (u_features - u_features.mean(axis=0)) / (u_features.std(axis=0) + 1e-10)
        u_matrix_features_norm = (u_matrix_features - u_matrix_features.mean()) / (u_matrix_features.std() + 1e-10)
        
        combined_features = np.concatenate([u_features_norm, u_matrix_features_norm], axis=1)
        labels_dual = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(combined_features)
        
        # 方法3: 仅转移矩阵特征
        logger.info("  Method 3: Matrix Feature Only (10-dim)...")
        labels_matrix = GaussianMixture(n_components=2, random_state=42, n_init=10).fit_predict(u_matrix_features_norm)
        
        # ==== Step 9: t-SNE可视化 ====
        logger.info(f"\n[Step 8] Performing t-SNE Visualization...")
        perplexity = min(30, max(5, len(u_indices) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(u_features)
        
        # ==== Step 10: 绘制对比图 ====
        logger.info(f"\n[Step 9] Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        clean_mask = u_labels == 0
        poison_mask = u_labels == 1
        
        # (0,0) 真实标签
        axes[0, 0].scatter(features_2d[clean_mask, 0], features_2d[clean_mask, 1],
                        c='green', label=f'Clean ({np.sum(clean_mask)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[0, 0].scatter(features_2d[poison_mask, 0], features_2d[poison_mask, 1],
                        c='red', label=f'Poisoned ({np.sum(poison_mask)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[0, 0].set_title('Ground Truth Labels', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('t-SNE Component 1')
        axes[0, 0].set_ylabel('t-SNE Component 2')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # (0,1) 标准特征聚类
        cluster_0 = labels_std == 0
        cluster_1 = labels_std == 1
        axes[0, 1].scatter(features_2d[cluster_0, 0], features_2d[cluster_0, 1],
                        c='blue', label=f'Cluster 0 ({np.sum(cluster_0)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[0, 1].scatter(features_2d[cluster_1, 0], features_2d[cluster_1, 1],
                        c='orange', label=f'Cluster 1 ({np.sum(cluster_1)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[0, 1].set_title('Method 1: Standard Features (512-dim)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('t-SNE Component 1')
        axes[0, 1].set_ylabel('t-SNE Component 2')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # (1,0) 双特征聚类
        cluster_0 = labels_dual == 0
        cluster_1 = labels_dual == 1
        axes[1, 0].scatter(features_2d[cluster_0, 0], features_2d[cluster_0, 1],
                        c='cyan', label=f'Cluster 0 ({np.sum(cluster_0)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[1, 0].scatter(features_2d[cluster_1, 0], features_2d[cluster_1, 1],
                        c='purple', label=f'Cluster 1 ({np.sum(cluster_1)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[1, 0].set_title('Method 2: Dual Features (512+10 dims)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE Component 1')
        axes[1, 0].set_ylabel('t-SNE Component 2')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # (1,1) 转移矩阵特征聚类
        cluster_0 = labels_matrix == 0
        cluster_1 = labels_matrix == 1
        axes[1, 1].scatter(features_2d[cluster_0, 0], features_2d[cluster_0, 1],
                        c='brown', label=f'Cluster 0 ({np.sum(cluster_0)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[1, 1].scatter(features_2d[cluster_1, 0], features_2d[cluster_1, 1],
                        c='pink', label=f'Cluster 1 ({np.sum(cluster_1)})', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
        axes[1, 1].set_title('Method 3: Matrix Features Only (10-dim)', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE Component 1')
        axes[1, 1].set_ylabel('t-SNE Component 2')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        set_name = "Intersection" if use_intersection else "Union"
        plt.suptitle(f'Clustering on P1 ∩ S1 ({set_name}): Target Class {predicted_target_class}, Size: {len(u_indices)}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        plot_path = os.path.join(self.args.save_path, 'kmeans_p1_s1_intersection.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   -> [Saved] Visualization to {plot_path}")
        
        # ==== Step 11: 评估 ====
        logger.info("\n" + "="*70)
        logger.info("[EVALUATION RESULTS]")
        logger.info("="*70)
    
    
        def align_and_evaluate_with_matrix_feature(pred_labels, true_labels, u_matrix_features, 
                                          u_indices, p1_indices, 
                                          method_name, logger=None):
            """
            pred_labels: 聚类标签 [U_size]
            true_labels: 真实毒性标签 [U_size] (0=clean, 1=poison)
            u_matrix_features: 矩阵特征 [U_size, num_classes] 或 [U_size]
            u_indices: U集合的索引 [U_size]
            p1_indices: P1集合的索引 [num_p1]
            method_name: 方法名称
            logger: 日志记录器
            """
            
            def log(msg):
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            
            # 计算P1内所有样本的索引集合（用于快速查询）
            p1_set = set(p1_indices)
            
            # 对于每个簇，找出既在该簇中又在P1中的样本
            cluster_0_mask = pred_labels == 0
            cluster_1_mask = pred_labels == 1
            
            # 簇0中同时在P1中的样本
            cluster_0_in_p1_mask = np.array([
                (cluster_0_mask[i] and u_indices[i] in p1_set) 
                for i in range(len(u_indices))
            ])
            
            # 簇1中同时在P1中的样本
            cluster_1_in_p1_mask = np.array([
                (cluster_1_mask[i] and u_indices[i] in p1_set) 
                for i in range(len(u_indices))
            ])
            
            # 计算两个簇的平均矩阵特征（仅考虑P1中的样本）
            if np.sum(cluster_0_in_p1_mask) > 0:
                cluster_0_avg_matrix_feature = np.mean(u_matrix_features[cluster_0_in_p1_mask])
                cluster_0_p1_count = np.sum(cluster_0_in_p1_mask)
            else:
                cluster_0_avg_matrix_feature = 0
                cluster_0_p1_count = 0
            
            if np.sum(cluster_1_in_p1_mask) > 0:
                cluster_1_avg_matrix_feature = np.mean(u_matrix_features[cluster_1_in_p1_mask])
                cluster_1_p1_count = np.sum(cluster_1_in_p1_mask)
            else:
                cluster_1_avg_matrix_feature = 0
                cluster_1_p1_count = 0
            
            # 判断哪个簇是毒性类（平均矩阵特征高的是毒性）
            poison_cluster = 0 if cluster_0_avg_matrix_feature >= cluster_1_avg_matrix_feature else 1
            
            # 将聚类标签对齐（毒性簇标记为1，干净簇标记为0）
            aligned_pred = np.where(pred_labels == poison_cluster, 1, 0)
            
            # 计算评估指标
            accuracy = accuracy_score(true_labels, aligned_pred)
            precision = precision_score(true_labels, aligned_pred, zero_division=0)
            recall = recall_score(true_labels, aligned_pred, zero_division=0)
            f1 = f1_score(true_labels, aligned_pred, zero_division=0)
            cm = confusion_matrix(true_labels, aligned_pred)
            
            log(f"\n{method_name}:")
            log(f"  [P1 Filtering Applied]")
            log(f"  Cluster 0: {np.sum(cluster_0_mask)} samples, {cluster_0_p1_count} in P1, avg matrix feature: {cluster_0_avg_matrix_feature:.4f}")
            log(f"  Cluster 1: {np.sum(cluster_1_mask)} samples, {cluster_1_p1_count} in P1, avg matrix feature: {cluster_1_avg_matrix_feature:.4f}")
            log(f"  → Cluster {poison_cluster} identified as Poisoned (higher avg matrix feature in P1)")
            log(f"  Accuracy:  {accuracy:.4f}")
            log(f"  Precision: {precision:.4f}")
            log(f"  Recall:    {recall:.4f}")
            log(f"  F1-Score:  {f1:.4f}")
            log(f"  Confusion Matrix:\n{cm}")
            
            return accuracy, precision, recall, f1
        
        # 提取目标列的非对角线和特征
        u_matrix_features_target_col = u_matrix_features[:, predicted_target_class]
        
        # 评估三种方法
        metrics_std = align_and_evaluate_with_matrix_feature(
            labels_std, u_labels, u_matrix_features_target_col,
            u_indices, p1_indices,
            "Method 1: Standard Features (512-dim)", logger=logger
        )
        
        metrics_dual = align_and_evaluate_with_matrix_feature(
            labels_dual, u_labels, u_matrix_features_target_col,
            u_indices, p1_indices,
            "Method 2: Dual Features (512+10 dims)", logger=logger
        )
        
        metrics_matrix = align_and_evaluate_with_matrix_feature(
            labels_matrix, u_labels, u_matrix_features_target_col,
            u_indices, p1_indices,
            "Method 3: Matrix Features Only (10-dim)", logger=logger
        )
        
        # 对比表
        logger.info("\n" + "="*70)
        logger.info("[SUMMARY COMPARISON]")
        logger.info("="*70)
        logger.info(f"{'Method':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        logger.info("-" * 88)
        logger.info(f"{'Std Features':<40} {metrics_std[0]:<12.4f} {metrics_std[1]:<12.4f} {metrics_std[2]:<12.4f} {metrics_std[3]:<12.4f}")
        logger.info(f"{'Dual Features':<40} {metrics_dual[0]:<12.4f} {metrics_dual[1]:<12.4f} {metrics_dual[2]:<12.4f} {metrics_dual[3]:<12.4f}")
        logger.info(f"{'Matrix Features Only':<40} {metrics_matrix[0]:<12.4f} {metrics_matrix[1]:<12.4f} {metrics_matrix[2]:<12.4f} {metrics_matrix[3]:<12.4f}")
        logger.info("="*88)
        
        # 计算改进
        logger.info("\n[IMPROVEMENT ANALYSIS]")
        logger.info("-" * 88)
        logger.info(f"Dual vs Std:     Recall +{(metrics_dual[2] - metrics_std[2])*100:.2f}pp, F1 +{(metrics_dual[3] - metrics_std[3])*100:.2f}pp")
        logger.info(f"Matrix vs Std:   Recall +{(metrics_matrix[2] - metrics_std[2])*100:.2f}pp, F1 +{(metrics_matrix[3] - metrics_std[3])*100:.2f}pp")
        logger.info("="*88 + "\n")

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
        
    trainer.visualize_kmeans_with_p1_s1_intersection()