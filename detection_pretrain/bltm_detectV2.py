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
from torchvision import transforms


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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # 产生与图像同维度的噪声
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

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
        
        # 使用 resnet.py 中的 ResNet18
        # 注意：resnet.py 默认 3通道输入，如果是 MNIST (1通道) 需要修改 resnet.py 或使用 ResNet18_F
        if self.args.input_channel == 1:
            print("Detected 1-channel input, using ResNet18_F...")
            model = resnet.ResNet18_F(self.args.num_classes)
        else:
            model = resnet.ResNet34(self.args.num_classes)
            
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
                
                # resnet.py 中的 forward 可能会返回 tuple (output, correction) 如果 revision=True
                # 这里我们默认 revision=False，只返回 output
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
                # 处理可能返回 tuple 的情况 (虽然 resnet.py 可能只返回 tensor)
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
        #train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train=True)
        
        # 基础的 Resize 和 ToTensor
        if self.args.dataset == 'cifar10':
            # 基于 CIFAR-10 的均值方差
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        else:
            # 默认
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        # 定义 "Hard Mode" 增强
        hard_train_tran = transforms.Compose([
            transforms.Resize((self.args.input_height, self.args.input_width)),
            transforms.RandomHorizontalFlip(), # 翻转增加一点难度
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            AddGaussianNoise(0., 0.2) # <--- 加入噪声！std越小，增强越轻微；std越大，增强越强烈
        ])
        
        
        
        bd_dataset = self.result['bd_train'].wrapped_dataset
        #bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=hard_train_tran, wrap_label_transform=None)
        
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
        # 这里的 num_classes 需要传入 C*C，因为 resnet_bayes.py 的 Linear 层定义是 out_features=num_classes
        t_net_output_dim = args.num_classes * args.num_classes
        
        if self.args.input_channel == 1:
            T_model = resnet_bayes.ResNet18_F(t_net_output_dim)
        else:
            T_model = resnet_bayes.ResNet34(t_net_output_dim)
        
        
        cls_state = classifier.state_dict()
        t_state = T_model.state_dict()
        
        # 过滤掉不匹配的层 (主要就是 'linear.weight' vs 'bayes_linear.weight')
        pretrained_dict = {k: v for k, v in cls_state.items() if k in t_state and v.size() == t_state[k].size()}
        t_state.update(pretrained_dict)
        T_model.load_state_dict(t_state)
        
        print(f"Initialized T-Net using Warm-up Classifier weights ({len(pretrained_dict)} layers matched).")

        T_model.to(self.device)
        T_model.train()

        # 5. 训练 T-Net
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
        if self.args.input_channel == 1:
            T_model = resnet_bayes.ResNet18_F(t_net_output_dim)
        else:
            T_model = resnet_bayes.ResNet34(t_net_output_dim)

        model_path = os.path.join(args.save_path, 'T_model_final.pth')
        if not os.path.exists(model_path):
            print("Model file not found, please train first.")
            return
        
        T_model.load_state_dict(torch.load(model_path, map_location=self.device))
        T_model.to(self.device)
        T_model.eval()
        
        print("###Loading Warm-up Classifier.....")
        if self.args.input_channel == 1:
                C_model = resnet.ResNet18_F(args.num_classes)
        else:
                C_model = resnet.ResNet34(args.num_classes)
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