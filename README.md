# TCPL-
复现 Frontiers in Neuroscience 2025 论文《TCPL: Task-Conditioned Prompt Learning for Few-shot Cross-subject Motor Imagery EEG Decoding》，实现基于任务条件提示（Task-Conditioned Prompt）与 TCN-Transformer 混合架构的跨被试脑电运动想象分类模型

## 工作内容
- 使用 MNE 完成 EEG 数据预处理与频段滤波
- 搭建 TCN-Transformer 混合时序建模结构
- 实现 Task-Conditioned Prompt 模块
- 构建 Few-shot 跨被试训练与测试流程
- 在 BCI Competition IV 2a 数据集上进行实验验证

## 项目亮点
- 理解跨被试 EEG 泛化问题
- 掌握 Prompt Learning 在脑机接口中的应用
- 实现 Few-shot EEG 解码流程
- 熟悉 Transformer 在时序脑电中的建模方法
