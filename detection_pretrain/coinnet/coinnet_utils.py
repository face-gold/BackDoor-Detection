import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ------------------------------------------------------------------
# Part 1: 影子模型定义 (ResNet)
# 我们在这里定义一个独立的 ResNet，以确保影子模型与主模型的架构或初始化不同
# ------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ------------------------------------------------------------------
# Part 2: 影子模型训练逻辑
# ------------------------------------------------------------------

class ShadowModelTrainer:
    def __init__(self, train_loader, device, num_classes, num_samples):
        self.loader = train_loader
        self.device = device
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        # 影子模型训练参数
        self.lr = 0.01 
        self.epochs = 25 # 20个epoch足够产生有效的噪声混淆矩阵
        self.weight_decay = 1e-4

    def train_single_model(self, seed_idx):
        """训练单个影子模型并返回预测结果"""
        # 设置随机种子以保证多样性
        torch.manual_seed(seed_idx)
        
        # 使用 ResNet34 结构 (不同于 ResNet18，增加多样性)
        model = ResNet(BasicBlock, [3, 4, 6, 3], self.num_classes)
        model = model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # 简单的 Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        print(f"[*] COINNet: Training Shadow Model #{seed_idx+1}...")
        
        # 1. 训练循环
        for epoch in range(self.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for batch_data in pbar:
                # BackdoorBench 默认返回 5 个值
                if len(batch_data) == 5:
                    data, target, _, _, _ = batch_data
                elif len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data[0], batch_data[1]
                
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # 2. 预测循环 (关键：使用 index 对齐)
        print(f"[*] COINNet: Generating predictions for Model #{seed_idx+1}...")
        all_preds = torch.zeros(self.num_samples, dtype=torch.long).to(self.device)
        
        model.eval()
        with torch.no_grad():
            for batch_data in self.loader:
                # 必须获取 index
                if len(batch_data) == 5:
                    data, _, index, _, _ = batch_data
                elif len(batch_data) == 3:
                    data, _, index = batch_data
                else:
                    raise ValueError("DataLoader must return indices to align crowd labels!")

                data = data.to(self.device)
                index = index.to(self.device)
                
                output = model(data)
                preds = torch.argmax(output, dim=1)
                
                # 填入正确的位置
                all_preds[index] = preds

        return all_preds.cpu()

def generate_crowd_labels(train_loader, device, num_classes, num_samples, num_annotators=3):
    """
    生成模拟众包标签
    """
    trainer = ShadowModelTrainer(train_loader, device, num_classes, num_samples)
    
    crowd_preds_list = []
    for m in range(num_annotators):
        # 使用不同的 m 作为种子
        preds = trainer.train_single_model(seed_idx=m)
        crowd_preds_list.append(preds)
    
    # 堆叠 -> [N, M]
    crowd_labels = torch.stack(crowd_preds_list, dim=1)
    return crowd_labels