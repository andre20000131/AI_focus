import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import os
import yaml
import torch
import matplotlib.pyplot as plt
from models.model import FourChannelResNet


from utils import calculate_psnr, calculate_ssim
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

# 加载配置
with open("config.yaml",encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建输出目录
os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["result_dir"], exist_ok=True)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FourChannelResNet().to(device)
optimizer = Adam(model.parameters(), lr=config["lr"])
criterion = MSELoss()

# 获取数据集
train_dataset, val_dataset = Dataset.get_train_val_datasets(
    config["train_face_dir"],
    config["train_tof_dir"],
    config["train_face_hr_dir"],
    config["train_tof_hr_dir"],
    config["train_gt_dir"],
    val_split=config["val_split"]
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 记录指标
history = {
    'train_loss': [],
    'val_loss': [],
    'psnr': [],
    'ssim': []
}

# 训练循环

for epoch in tqdm(range(config["epochs"]),desc=f"Epoch/{config['epochs']} [Train]"):
    # 训练阶段
    model.train()
    epoch_train_loss = 0.0
    for lr, gt in train_loader:
     #   print(lr.shape,hr.shape)
        lr, gt = lr.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, gt)
        #print(output,gt)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * lr.size(0)
    train_loss = epoch_train_loss / len(train_loader.dataset)
    history['train_loss'].append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)

            # 计算损失
            val_loss += criterion(output, hr).item()



    # 记录验证指标
    history['val_loss'].append(val_loss / len(val_loader))


    # 打印进度
    print(f"Epoch {epoch + 1}/{config['epochs']}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {history['val_loss'][-1]:.4f}")


    # 保存模型
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), f"{config['save_dir']}/model_epoch_{epoch + 1}.pth")

# 保存训练曲线
plt.figure(figsize=(12, 8))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()



plt.tight_layout()
plt.savefig(f"{config['result_dir']}/training_metrics.png")
plt.close()