import torch
import yaml
from models.model import FourChannelResNet
from datasets import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

# 加载配置和模型
with open("config.yaml",encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = FourChannelResNet()
model.load_state_dict(torch.load("./train_res/checkpoints/model_epoch_90.pth",weights_only=True))
model.eval()

# 测试数据集
test_dataset = Dataset(config["val_face_dir"], config["val_tof_dir"],config["val_face_hr_dir"],config["val_tof_hr_dir"],config["val_gt_dir"], patch_size=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 测试并保存结果
with torch.no_grad():
    for i, (lr, hr,filename) in enumerate(test_loader):
        #print(hr)
        output = model(lr)
        output = output.numpy() * 255
        hr = hr.numpy() * 255

        output = output.astype(np.uint8)
        hr = hr.astype(np.uint8)
        print(filename,output,hr)