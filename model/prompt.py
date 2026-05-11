import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportEncoder(nn.Module):
    """编码器：独立的 2 层 1D 卷积网络，用于支撑集的表示提取。
    
    论文规定：独立网络，不与 TCN 共享参数。
    2 层 1D 卷积 (kernel=3, stride=1, same padding) -> 全局平均池化 -> 线性投影至 d。
    """
    def __init__(self, in_channels, d=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, d, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(d)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(d)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d, d)

    def encode(self, x):
        """x: (B, C, T) → (B, d)"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x).squeeze(-1)  # (B, d)
        x = self.proj(x)                       # (B, d)
        return x

    def forward(self, x):
        return self.encode(x)


class PromptGenerator(nn.Module):
    """Prompt 生成器。
    
    论文规定：对一个 subject 所有支撑嵌入取平均，得到单个向量 hi。
    送入两层 MLP (隐藏层宽度 d, ReLU) 输出 k*d 并 reshaep 为 k 个 prompt tokens。
    """
    def __init__(self, d=64, k=10):
        super().__init__()
        self.k = k
        self.d = d

        # 由单个向量 hi 映射到 k * d 维度的 prompts
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, k * d)
        )

    def forward(self, h_i):
        """h_i: (1, d) 或 (d,) → prompts: (k, d)"""
        prompts_flat = self.mlp(h_i)               # (1, k*d) or (k*d)
        generated = prompts_flat.view(self.k, self.d)  # (k, d)
        return generated


class TCPModule(nn.Module):
    """任务条件提示模块"""
    def __init__(self, in_channels, d=64, k=10):
        super().__init__()
        self.encoder = SupportEncoder(in_channels, d)
        self.generator = PromptGenerator(d, k)

    def forward(self, support_set, support_labels=None):
        """从支持集生成 prompts。
        
        support_set: (N_s, C, T)
        返回 prompts: (k, d)
        """
        embeddings = self.encoder.encode(support_set)          # (N_s, d)
        h_i = embeddings.mean(dim=0, keepdim=True)             # (1, d)
        prompts = self.generator(h_i)                          # (k, d)
        return prompts
