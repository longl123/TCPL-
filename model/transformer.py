import torch  # 导入PyTorch机器学习基础包
import torch.nn as nn  # 导入PyTorch的神经网络图层及构建块体系
import math  # 导入Python原生数学运算支持库，用来作各种基础的特定数学求解(如正弦余弦、对数)

class PositionalEncoding(nn.Module):  # 定义序列的位置编码添加组件，以便给不含有时序先后的注意力系统提示时序的前后顺次关系
    def __init__(self, d_model, max_len=2000):  # 类实例化初始化：要求输入的向量内部纬度数目大小d_model和可能涵盖应对的最繁长序列尺寸max_len
        super().__init__()  # 先让底层大框架把自身底层准备完备进行初始启动安排
        pe = torch.zeros(max_len, d_model)  # 创建一块先以全部数字填充为全零用来装填固定长度编码结果的大展板矩阵(二维：序列长 x 特征维)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # 排列生产出一条步长位置下标增量数组（形如[0,1,2...,max_len]），变更为立柱形态作为每一项的次序数用于乘法基准
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 根据偶数下标步步衰减算法提前准备算出Transformer原论述要求用来分割周期缩放长宽频率波的特有指数衰减常量值阵列
        pe[:, 0::2] = torch.sin(position * div_term)  # 利用以上推算针对这块大板子上全体偶分段索引行去计算对应着各自时间前后的特定的正弦频率三角波分量来打上空间标记
        pe[:, 1::2] = torch.cos(position * div_term)  # 与上面动作互辅而行将全部余下归属奇数段上的行对应填入错位角度周期的特定的基于推算出的余弦三角波纹印记予以编组辅助标定
        self.register_buffer('pe', pe)  # 调用缓存声明登记命令让该非可求导变量(无需学习训练)能够依附绑牢挂到网络保存架构内一块被归在自动计算和转移内而不加入求梯度的数组内随整体走

    def forward(self, x):  # 这个单独块针对传递过来的单方面串流实施执行编码装载嵌入的正向注入业务流程
        # x: (batch, seq_len, d_model)  # 提供解说表明外部馈送进函数阵的矩阵具有此番三级结构顺序规范: [分批数量、时序号深浅数、映射厚度]
        return x + self.pe[:x.size(1)]  # 原数据沿最后时序号只取对应有效序列位大小宽的印记板直接覆盖重压重加贴于自身矩阵中以此叠加时序编码的效应


class TransformerEncoder(nn.Module):  # 开始搭构全局性质统合串连并挖掘多点关联的主构自注意力核心组件——Transformer编纂机
    def __init__(self, d_model=64, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):  # 送配置启动包含：基维大跨度、并行查探的源头头数目数、编解码复奏多深厚轮次，和隐藏变向扩张时尺寸以及容错遗忘退出掉落率
        super().__init__()  # 首先触发底层主类装设
        self.pos_encoder = PositionalEncoding(d_model)  # 例化上文中撰写规划停当前以加入各序列时基印记刻花功能的时序定点层组件并纳归持有
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,  # 指定要求呼叫标准的单独Transformer编排大层生成命令，按输入要求配置设定模型基础深广特征及内部前串机制的大小规格量纲
                                                   dropout, batch_first=True, norm_first=True)  # 使用 norm_first=True (Pre-LN) 避免无 warmup 时的梯度消失和模型崩溃
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)  # 正式调集搭建堆叠组装成完整的庞大变送中枢骨干模块把方才单个编组根据输入指令铺放堆压多次形成层叠机制

    def forward(self, x, src_key_padding_mask=None):  # 设计让目标矩阵进站通行流经全体机工组合完成全视野自我关联理解加工正向流传规则
        # x: (batch, seq_len, d_model)  已包含提示 token  # 说解送至加工中转线的全态内容(已经接上最头部的数条专有指令信号在这一股队列包一起)的模样
        x = self.pos_encoder(x)  # 第一个动作将光板序列压入我们设的打时标码机床去为所有的排列贴标签从而注入每个特异的时间占位记忆
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # 将打上记号标的条块再整个推送至重磅关注汇聚编排流水线上依照需求将注意力机制计算掩码避开空白进行统聚运算
        return out  # 回送吐出经过全局目光关联整合完毕并携带有互相纠结融会的高级认识特征最终条列产物