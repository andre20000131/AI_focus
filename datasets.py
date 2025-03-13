import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, random_split
class Dataset(Dataset):
    def __init__(self, face, tof,face_hr,tof_hr,gt_txt_path,scale_factor=4, patch_size=96):
        self.face_hr = face_hr
        self.tof_hr = tof_hr
        self.face = face
        self.tof = tof
        self.scale = scale_factor
        self.patch_size = patch_size
        self.gt_dict = self._load_gt_values(gt_txt_path)
        self.filenames = os.listdir(face)




    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # 构建路径
        lr_path = os.path.join(self.face, filename)
        hr_path = os.path.join(self.tof, filename)
        face_hr = os.path.join(self.face_hr, filename)
        tof_hr = os.path.join(self.tof_hr, filename)
        # 读取图像并转换为 RGB

        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        face_hr_img = cv2.imread(face_hr)
        tof_hr_img = cv2.imread(tof_hr)

        lr_img = cv2.resize(lr_img,(480,270))
        hr_img = cv2.resize(hr_img,(480,270))
        face_hr_img = cv2.resize(face_hr_img,(480,270))
        tof_hr_img = cv2.resize(tof_hr_img,(480,270))


        # 转换为 RGB 格式
        #print(lr_img)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        face_hr_img = cv2.cvtColor(face_hr_img, cv2.COLOR_BGR2RGB)
        tof_hr_img = cv2.cvtColor(tof_hr_img, cv2.COLOR_BGR2RGB)



        # 将第二张RGB转为灰度图（单通道）
        hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_RGB2GRAY)  # 转为单通道

        # 归一化处理
        lr_patch = lr_img.astype(np.float32) / 255.0
        hr_patch = hr_gray.astype(np.float32) / 255.0
        face_hr_patch = face_hr_img.astype(np.float32) / 255.0
        tof_hr_patch = tof_hr_img.astype(np.float32) / 255.0


        # 转换为Tensor并调整维度
        lr_patch = torch.from_numpy(lr_patch).permute(2, 0, 1)  # [3, H, W]
        face_hr_patch = torch.from_numpy(face_hr_patch).permute(2, 0, 1)  # [3, H, W]
        tof_hr_patch = torch.from_numpy(tof_hr_patch).permute(2, 0, 1)  # [3, H, W]


        hr_patch = torch.from_numpy(hr_patch).unsqueeze(0)  # [1, H, W] 添加通道维度




        # 合并为四通道
        combined = torch.cat((face_hr_patch,tof_hr_patch,lr_patch, hr_patch), dim=0)  # [10, H, W]
        gt = self.gt_dict.get(filename, 0.0)  # 如果找不到文件名，默认返回0.0
        gt = torch.tensor(gt, dtype=torch.float32)  # 将gt值转为Tensor
        gt = gt/255
        return combined, gt,filename

    def get_train_val_datasets(face, tof,face_hr,tof_hr,gt_txt_path,val_split=0.2):
        full_dataset = Dataset(face, tof,face_hr,tof_hr,gt_txt_path)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        return train_dataset, val_dataset

    def _load_gt_values(self, gt_txt_path):
        """
        从txt文件加载gt值到字典
        :param gt_txt_path: txt文件路径
        :return: 字典 {filename: gt_value}
        """
        gt_dict = {}
        with open(gt_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:  # 确保每行是文件名和gt值
                    filename, gt_value = parts
                    gt_dict[filename] = float(gt_value)  # 假设gt值是浮点数

        return gt_dict