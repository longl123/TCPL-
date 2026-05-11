import torch
import torch.nn as nn

class DilatedConv1d(nn.Module):
    """标准的非对称因果膨胀卷积，满足论文只能利用过去序列的要求"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        # 裁剪掉右侧多余的 padding (未来信息)，实现因果卷积约束
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """残差 TCN 块，包含两个因果膨胀卷积"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = DilatedConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = DilatedConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return self.relu2(out + residual)


class TCN(nn.Module):
    """时序卷积网络：完全复现原文，直接将包含因果约束的膨胀卷积层应用在特征处理后的脑电上，无额外空间融合池化下采样动作。
    输出 (T, d) 序列
    """
    def __init__(self, in_channels, kernel_size=3, dilations=(1, 2, 4, 8),
                 channels=(64, 64, 128, 128), dropout=0.1, d=64):
        super().__init__()

        layers = []
        in_ch = in_channels
        for out_ch, dil in zip(channels, dilations):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dil, dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv1d(channels[-1], d, 1)

    def forward(self, x):
        """x: (batch, C, T) → (batch, T, d)"""
        x = self.net(x)                                    # (B, out_ch, T)
        x = self.proj(x)                                   # (B, d, T)
        x = x.transpose(1, 2)                              # (B, T, d)
        return x
