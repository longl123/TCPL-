import torch  
import torch.nn as nn  
import math 

class PositionalEncoding(nn.Module):  # 定义序列的位置编码添加组件，以便给不含有时序先后的注意力系统提示时序的前后顺次关系
    def __init__(self, d_model, max_len=2000):  # 类实例化初始化：要求输入的向量内部纬度数目大小d_model和可能涵盖应对的最繁长序列尺寸max_len
        super().__init__()  
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1).float()  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  # 这个单独块针对传递过来的单方面串流实施执行编码装载嵌入的正向注入业务流程
        # x: (batch, seq_len, d_model)  
        return x + self.pe[:x.size(1)]  

class TransformerEncoder(nn.Module):  # 开始搭构全局性质统合串连并挖掘多点关联的主构自注意力核心组件——Transformer编纂机
    def __init__(self, d_model=64, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):  # 送配置启动包含：基维大跨度、并行查探的源头头数目数、编解码复奏多深厚轮次，和隐藏变向扩张时尺寸以及容错遗忘退出掉落率
        super().__init__()  
        self.pos_encoder = PositionalEncoding(d_model)  
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
                                                   dropout, batch_first=True, norm_first=True)  
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers) 
        
    def forward(self, x, src_key_padding_mask=None):  # 设计让目标矩阵进站通行流经全体机工组合完成全视野自我关联理解加工正向流传规则
        # x: (batch, seq_len, d_model)  已包含提示 token  
        x = self.pos_encoder(x)  
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask) 
        return out 
