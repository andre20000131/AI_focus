import torch
from torch import nn


class FourChannelResNet(nn.Module):
    def __init__(self):
        super(FourChannelResNet, self).__init__()

        # 输入处理 (4通道适配)
        self.conv1 = nn.Conv2d(10, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet基本块定义
        # Block 1
        self.block1_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block1_bn1 = nn.BatchNorm2d(256)
        self.block1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(256)

        # Block 2
        self.block2_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block2_bn1 = nn.BatchNorm2d(256)
        self.block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block2_bn2 = nn.BatchNorm2d(256)

        # 下采样层
        self.downsample_conv = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False)
        self.downsample_bn = nn.BatchNorm2d(256)

        # Block 3 (带下采样)
        self.block3_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.block3_bn1 = nn.BatchNorm2d(256)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block3_bn2 = nn.BatchNorm2d(256)

        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.block4_bn1 = nn.BatchNorm2d(256)
        self.block4_conv2 = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=False)
        self.block4_bn2 = nn.BatchNorm2d(256)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(2040, 128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # 初始处理
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Block 1
        identity = x
        x = self.block1_conv1(x)
       # x = self.block1_bn1(x)
        x = self.relu(x)
        x = self.block1_conv2(x)
       # x = self.block1_bn2(x)
        x += identity
        x = self.relu(x)

        # Block 2
        identity = x
        x = self.block2_conv1(x)
      #  x = self.block2_bn1(x)
        x = self.relu(x)
        x = self.block2_conv2(x)
       # x = self.block2_bn2(x)
        x += identity
        x = self.relu(x)

        # 下采样路径
        identity = self.downsample_conv(x)
       # identity = self.downsample_bn(identity)

        # Block 3
        x = self.block3_conv1(x)
       # x = self.block3_bn1(x)
        x = self.relu(x)
        x = self.block3_conv2(x)
       # x = self.block3_bn2(x)
        x += identity
        x = self.relu(x)

        # Block 4
        identity = x
        x = self.block4_conv1(x)
       # x = self.block4_bn1(x)
        x = self.relu(x)
        x = self.block4_conv2(x)
        #x = self.block4_bn2(x)
       # x += identity
        x = self.relu(x)

        # 输出处理
        #print(x.shape)
        #x = self.avgpool(x)
       # print(x.shape)
        x = torch.flatten(x, 1)
       # print(x.shape)
        x = self.fc1(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # 测试网络结构
    model = FourChannelResNet()
    test_input = torch.randn(2, 4, 480, 370)  # (batch, channels, height, width)
    output = model(test_input)
    print(f"输入尺寸：{test_input.shape}")
    print(f"输出尺寸：{output.shape}")  # 应该输出 torch.Size([2, 1])

