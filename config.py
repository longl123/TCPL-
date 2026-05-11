"""BCI2a + TCPL 模型超参数配置"""

import torch


class Config:
    # ==================== 数据集路径 ====================
    data_dir = r"D:\下载\BCICIV_2a_gdf"

    # ==================== 数据参数 ====================
    n_channels = 22
    n_times = 512
    n_classes = 4

    # ==================== TCP 提示 ====================
    k_prompts = 10
    prompt_dim = 64

    # ==================== TCN ====================
    tcn_kernel = 3
    tcn_dilations = [1, 2, 4, 8]
    tcn_channels = [64, 64, 128, 128]
    tcn_dropout = 0.1

    # ==================== Transformer ====================
    # 论文设定
    tr_layers = 4
    tr_heads = 8
    tr_dropout = 0.1
    tr_ffn_dim = 256

    # ==================== 元训练 (few-shot) ====================
    lr = 1e-3
    weight_decay = 1e-4
    batch_size_episodes = 16           
    epochs = 40
    shots = 10
    query_samples = 15
    lambda_reg = 1e-4
    tasks_per_subject = 5

    # ==================== 评估 ====================
    eval_support_sets = 5             

    # ==================== 设备与种子 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
