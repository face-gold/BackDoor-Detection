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

sys.path.append('../')
sys.path.append(os.getcwd())

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.aggregate_block.fix_random import fix_random
from tqdm import tqdm

from defense.base import defense
from utils.save_load_attack import load_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_transform
from utils.bd_dataset_v2 import dataset_wrapper_with_transform
import torchattacks
from man.resnet_T import ResNet18


def craft_adversarial_example(model, x_natural, y, step_size=2/255, epsilon=8/255, perturb_steps=10):
    # PGD 攻击
    attack = torchattacks.PGD(model, eps=epsilon, alpha=step_size, steps=perturb_steps, random_start=True)
    
    x_adv = attack(x_natural, y)
    return x_adv

def craft_adversarial_example_spe(classifier, T_model, x_natural, x_adv, step_size=2 / 255, epsilon=8 / 255, perturb_steps=10):
    # 针对 T 网络的特殊攻击 (Consistency Regularization)
    x_adv_spe = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv_spe.requires_grad_()
        with torch.enable_grad():
            logits_adv = classifier(x_adv_spe)
            outputs = F.softmax(logits_adv, dim=1)
            _, y_adv = torch.max(outputs.data, 1)

            T_adv = T_model(x_adv)
            T_adv_spe = T_model(x_adv_spe)

            loss_1 = F.cross_entropy(logits_adv, y_adv)
            loss_2 = nn.MSELoss()(T_adv_spe, T_adv)

            # 目标：最大化 T 的差异，同时保持分类预测不变(或变化)
            loss = loss_2 - loss_1

        grad = torch.autograd.grad(loss, [x_adv_spe])[0]
        x_adv_spe = x_adv_spe.detach() + step_size * torch.sign(grad.detach())
        x_adv_spe = torch.min(torch.max(x_adv_spe, x_natural - epsilon), x_natural + epsilon)
        x_adv_spe = torch.clamp(x_adv_spe, 0.0, 1.0)

    return x_adv_spe


def standard_loss(classifier, T_model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10):

    classifier.eval()
    T_model.eval()

    # 1. 生成 PGD 对抗样本 (在中毒样本基础上叠加对抗扰动)
    x_adv = craft_adversarial_example(classifier, x_natural, y, step_size=step_size, epsilon=epsilon,
                                      perturb_steps=perturb_steps)

    # 2. 生成特殊扰动样本
    x_adv_spe = craft_adversarial_example_spe(classifier, T_model, x_natural, x_adv, step_size=step_size, epsilon=epsilon,
                                              perturb_steps=perturb_steps)

    T_model.train()
    optimizer.zero_grad()

    # --- A. 自然样本 Loss ---
    logits = classifier(x_natural)
    T_pre = T_model(x_natural) 
    pred_labels = F.softmax(logits, dim=1)
    
    # 确保 T_pre 形状正确 [B, 10, 10]
    if T_pre.dim() == 2: T_pre = T_pre.view(-1, 10, 10)
    
    noisy_post = torch.bmm(pred_labels.unsqueeze(1), T_pre).squeeze(1)
    logits_nat = torch.log(noisy_post + 1e-12)
    loss_nat = nn.NLLLoss()(logits_nat, y)

    # --- B. PGD对抗样本 Loss ---
    logits = classifier(x_adv)
    T_adv = T_model(x_adv)
    pred_labels = F.softmax(logits, dim=1)
    
    if T_adv.dim() == 2: T_adv = T_adv.view(-1, 10, 10)
    
    noisy_post = torch.bmm(pred_labels.unsqueeze(1), T_adv).squeeze(1)
    logits_adv = torch.log(noisy_post + 1e-12)
    loss_adv = nn.NLLLoss()(logits_adv, y)

    # --- C. 稳定性 Loss ---
    T_spe = T_model(x_adv_spe)
    loss_T = nn.MSELoss()(T_spe, T_adv) # 强迫 T(x_adv) 和 T(x_spe) 一致

    # 总 Loss
    loss = loss_adv + 0.1 * loss_nat + 700.0 * loss_T

    return loss

class MANTrainer(defense):
    
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
        parser.add_argument('--amp', default = False, type=lambda x: str(x) in ['True','true','1'])
        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result (folder name in record/)')
    
        # 训练控制参数
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/detection/man_t/cifar10.yaml", help='the path of yaml')
        parser.add_argument('--sgd_momentum', type=float, help='sgd momentum')
        parser.add_argument('--sgd_weight_decay', type=float, help='sgd weight decay')

        # --- MAN 特有参数 ---
        parser.add_argument('--epsilon', default=8/255, type=float, help='perturbation')
        parser.add_argument('--num-steps', default=10, type=int, help='perturb number of steps')
        parser.add_argument('--step-size', default=2/255, type=float, help='perturb step size')
        parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/detection/man_t/'
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
        self.device = self.args.device

    def train(self):
        """
        主训练流程
        """
        print("===> Starting Process...")
        self.set_devices()
        fix_random(self.args.random_seed)
        self.set_logger()
        
        args = self.args
        device = torch.device(args.device)
        
        classifier = generate_cls_model(self.args.model, args.num_classes)
        classifier.load_state_dict(self.result['model'])
        

        if "," in self.device:
            classifier = torch.nn.DataParallel(
                classifier,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{classifier.device_ids[0]}'
            classifier.to(self.args.device)
            classifier.eval()
        else:
            classifier.to(self.args.device)
            classifier.eval()
            
        #logging.info("==> Backdoored Classifier Loaded and Frozen.")
        print("==> Backdoored Classifier Loaded and Frozen.")

        # 2. 准备数据
        
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        bd_dataset = self.result['bd_train'].wrapped_dataset
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    
        #logging.info(f"==> Poisoned Training Data Loaded. Size: {len(bd_train_dataset)}")
        print(f"==> Poisoned Training Data Loaded. Size: {len(bd_train_dataset)}")

        # 3. 初始化 T 网络
        #logging.info("==> Initializing Transition Network (T-Net)...")
        print("==> Initializing Transition Network (T-Net)...")
        # CIFAR10: 10类 -> 输出 100
        t_net_output_dim = args.num_classes * args.num_classes
        T_model = ResNet18(t_net_output_dim)
        if "," in self.device:
            T_model = torch.nn.DataParallel(
                T_model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            T_model.to(self.args.device)
        else:
            T_model.to(self.args.device)

     
        # 4. 优化器
        optimizer = optim.SGD(T_model.parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.sgd_weight_decay)

        # 5. 训练循环
        #logging.info("==> Start Training...")
        print("==> Start Training...")
        start_time = time.time()
        
        
        for epoch in range(1, args.epochs + 1):
            
            if epoch >= 0.75 * args.epochs:
                lr = args.lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif epoch >= 0.9 * args.epochs:
                lr = args.lr * 0.01
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            total_loss = 0
            classifier.eval()
            
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")
            for batch_idx, (data, target, *others) in pbar:
                data, target = data.to(device), target.to(device)
                
                loss = standard_loss(
                    classifier=classifier,
                    T_model=T_model,
                    x_natural=data,  
                    y=target,        
                    optimizer=optimizer,
                    step_size=args.step_size,
                    epsilon=args.epsilon,
                    perturb_steps=args.num_steps
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % args.log_interval == 0:
                    pbar.set_postfix(loss=loss.item())
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

            avg_loss = total_loss / len(train_loader)
            logging.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
            
            # 保存 Checkpoint
            if epoch % 5 == 0 or epoch == args.epochs:
                 save_file = os.path.join(args.save_path, f'T_model_epoch_{epoch}.pth')
                 torch.save(getattr(T_model, 'module', T_model).state_dict(), save_file)
                 logging.info(f"Saved checkpoint to {save_file}")

        end_time = time.time()
        print(f"Total training time: {(end_time - start_time)/60:.2f} mins")
        #logging.info(f"Total training time: {(end_time - start_time)/60:.2f} mins")
        
        # 保存最终模型
        final_save_file = os.path.join(args.save_path, f'T_model_final.pth')
        torch.save(getattr(T_model, 'module', T_model).state_dict(), final_save_file)
        print(f"Saved final T model to {final_save_file}")
        #logging.info(f"Saved final T model to {final_save_file}")
    
    def detection(self):
        args = self.args
        device = torch.device(args.device)
        # 加载训练好的 T_model
        t_net_output_dim = args.num_classes * args.num_classes
        T_model = ResNet18(t_net_output_dim)
        model_path = os.path.join(args.save_path, f'T_model_final.pth')
        state_dict = torch.load(model_path, map_location=device)
        T_model.load_state_dict(state_dict)
        T_model.to(device)
        T_model.eval()
        
        train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
        bd_dataset = self.result['bd_train'].wrapped_dataset
        poison_indicator = np.array(bd_dataset.poison_indicator)
        bd_train_dataset = dataset_wrapper_with_transform(bd_dataset, wrap_img_transform=train_tran, wrap_label_transform=None)
        train_loader = DataLoader(
            bd_train_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,  # 检测时不要打乱，方便索引对应
            num_workers=int(args.num_workers),
            pin_memory=args.pin_memory
        )
        
        all_T = []
        with torch.no_grad():
            for batch_idx, (data, target, *_) in enumerate(train_loader):
                data = data.to(device)
                T_pred = T_model(data)
                T_pred = T_pred.view(-1, args.num_classes, args.num_classes).cpu().numpy()
                all_T.append(T_pred)
        all_T = np.concatenate(all_T, axis=0)  # [N, K, K]
        all_index = np.arange(len(all_T))
        all_is_poison = poison_indicator[:len(all_T)]
        
        # 保存到 npz
        npz_path = os.path.join(args.save_path, "all_T_matrix.npz")
        np.savez(npz_path, all_index=all_index, all_is_poison=all_is_poison, all_T=all_T)
        print(f"Saved all sample transition matrices to {npz_path}")

        # 保存到 all_T_matrix.txt
        txt_path = os.path.join(args.save_path, "all_T_matrix.txt")
        with open(txt_path, "w") as f:
            for idx, is_p, T in zip(all_index, all_is_poison, all_T):
                f.write(f"Index: {idx}\n")
                f.write(f"Is_Poison: {is_p}\n")
                f.write("Transition Matrix:\n")
                for row in T:
                    f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
                f.write("-------------------\n")
            # 统计所有转移矩阵之和
            total_T = np.sum(all_T, axis=0)  # [K, K]
            f.write("Total Transition Matrix (Sum):\n")
            for row in total_T:
                f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
            f.write("-------------------\n")
            # 归一化（每列归一化）
            norm_T = total_T / (np.sum(total_T, axis=0, keepdims=True) + 1e-12)
            f.write("Total Transition Matrix (Column Normalized):\n")
            for row in norm_T:
                f.write("\t".join([f"{v:.6f}" for v in row]) + "\n")
            f.write("-------------------\n")
        print(f"Saved all sample transition matrices and summary to {txt_path}")
        
        

# ==========================================
# Part 3: 主程序入口
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MAN T-Net on Backdoored Data')  
    MANTrainer.add_arguments(parser)
    args = parser.parse_args()

    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
        print(f"Warning: result_file not specified. Using default: {args.result_file}")

    trainer = MANTrainer(args)
    # 判断是否执行过训练
    t_model_files = [f for f in os.listdir(trainer.args.save_path) if f.startswith('T_model') and f.endswith('.pth')]
    if len(t_model_files) > 0:
        print("检测到已训练模型，跳过训练，直接进入检测。")
        trainer.detection()
    else:
        trainer.train()
        trainer.detection()
    