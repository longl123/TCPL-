import torch
import torch.nn as nn
import torch.nn.functional as F
from models.prompt import TCPModule
from models.tcn import TCN
from models.transformer import TransformerEncoder


class TCPL(nn.Module):
    """TCPL 模型 (论文严格复现版本)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.prompt_dim
        k = config.k_prompts
        n_cls = config.n_classes

        # TCN（独立）
        self.tcn = TCN(config.n_channels, config.tcn_kernel, config.tcn_dilations,
                       config.tcn_channels, config.tcn_dropout, d)

        # TCP 模块（独立，不再分享 TCN 的 temporal_spatial 或 pool）
        self.tcp = TCPModule(config.n_channels, d, k)

        # Transformer 恢复为严格参数配置
        self.transformer = TransformerEncoder(d, config.tr_heads,
                                              config.tr_layers, config.tr_ffn_dim,
                                              config.tr_dropout)
        self.classifier = nn.Linear(d, n_cls)


    def forward(self, x, prompts):
        """x: 查询样本 (B, C, T), prompts: (k, d) → logits: (B, n_classes)"""
        batch_size = x.size(0)
        h = self.tcn(x)                                          # (B, T', d)
        prompts_expanded = prompts.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, k, d)
        z = torch.cat([prompts_expanded, h], dim=1)              # (B, k+T', d)
        z_out = self.transformer(z)                              # (B, k+T', d)
        prompt_out = z_out[:, :self.config.k_prompts, :].mean(dim=1)  # (B, d)
        logits = self.classifier(prompt_out)
        return logits

    def meta_train_step(self, support_set, query_data, support_labels=None, **kwargs):
        """训练时前向：生成 prompt。
        返回:
            logits_tcpl: (B, n_classes) TCPL 分支预测
            prompts: (k, d) 用于正则化
        """
        prompts = self.tcp(support_set, support_labels)           # (k, d)
        logits_tcpl = self.forward(query_data, prompts)
        return logits_tcpl, prompts

    def predict_by_support(self, support_set, query_data, support_labels=None):
        """测试时推理：返回预测和 logits"""
        with torch.no_grad():
            prompts = self.tcp(support_set, support_labels)
            logits = self.forward(query_data, prompts)
            preds = logits.argmax(dim=1)
        return preds, logits
