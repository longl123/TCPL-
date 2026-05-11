"""
BCI Competition IV 2a (BCI2a) 数据集加载与预处理.

数据集概览:
- 9 名受试者 (A01–A09), 4 类运动想象: 左手(0), 右手(1), 双脚(2), 舌头(3)
- 每名受试者有 2 个 session (不同日期):
  - T session (AxxT.gdf): 训练 session
  - E session (AxxE.gdf): 评估 session
- 每个 session = 6 runs × 48 trials = 288 trials (每类 72, 均匀随机分布)

单次 trial 时序:
  t=0s:  十字注视点出现 + 提示音
  t=2s:  箭头提示 (←左手 / →右手 / ↓双脚 / ↑舌头), 持续 1.25s
  t=6s:  十字消失, MI 结束, 短暂休息 (~8s/trial)

GDF 事件码:
  768 = trial 开始, 769 = 左手(0), 770 = 右手(1), 771 = 双脚(2), 772 = 舌头(3)
  783 = 通用标记 (部分 E 文件中替代 769–772, 无类别区分能力)
"""

import numpy as np  
import torch  
from torch.utils.data import Dataset, Subset  
import mne  
from mne.io import read_raw_gdf  


class EEGSubjectDataset(Dataset):  # 定义单个受试者的EEG数据集类，继承自PyTorch的Dataset
    """单个受试者的 EEG 数据集.
    data:  (N, C, T) float tensor
    labels: (N,) long tensor — 0/1/2/3
    """
    def __init__(self, data, labels):  # 初始化方法，接收EEG数据和对应的标签
        super().__init__() 
        self.data = torch.from_numpy(data).float() 
        self.labels = torch.from_numpy(labels).long()  

    def __len__(self):  # 返回数据集中样本总数的方法
        return len(self.data)  

    def __getitem__(self, idx):  # 根据索引获取数据集中单个样本及其标签的方法
        return self.data[idx], self.labels[idx] 

def sample_task(subject_dataset, shots, q_per_class):  # 定义从受试者数据中采样小样本任务（支持集和查询集）的函数
    """从单个受试者数据中随机采样一个 meta-learning 任务 (支持集+查询集).

    支持集和查询集在每个类别内不重叠.
    返回 (support_set, query_set) — 两个 torch.utils.data.Subset.
    """
    data = subject_dataset.data  
    labels = subject_dataset.labels  
    n_classes = labels.max().item() + 1  

    support_indices = []  
    query_indices = [] 
    for c in range(n_classes): 
        idx_c = (labels == c).nonzero(as_tuple=False).squeeze() 
        perm = torch.randperm(len(idx_c))  
        idx_c = idx_c[perm] 
        n_support = min(shots, len(idx_c))  
        n_query = min(q_per_class, len(idx_c) - n_support)  
        support_indices.extend(idx_c[:n_support].tolist())  
        query_indices.extend(idx_c[n_support:n_support + n_query].tolist())  

    support_set = Subset(subject_dataset, support_indices)  
    query_set = Subset(subject_dataset, query_indices)  
    return support_set, query_set  


def sample_test_task(subject_dataset, shots):  # 定义为测试流程采样的函数: 指定数量的支持集，剩下的全为查询集
    """为测试受试者采样任务: 固定 shots 个支持样本, 其余全部作为查询集.
    返回 (support_set, query_set).
    """
    data = subject_dataset.data  
    labels = subject_dataset.labels  
    n_classes = labels.max().item() + 1  

    support_indices = []  
    query_indices = []  
    for c in range(n_classes):  
        idx_c = (labels == c).nonzero(as_tuple=False).squeeze()  
        perm = torch.randperm(len(idx_c))  
        idx_c = idx_c[perm]  
        n_support = min(shots, len(idx_c))  
        support_indices.extend(idx_c[:n_support].tolist())  
        query_indices.extend(idx_c[n_support:].tolist())  

    support_set = Subset(subject_dataset, support_indices)  
    query_set = Subset(subject_dataset, query_indices)  
    return support_set, query_set  


# --------------------- BCI2a 预处理与加载 ---------------------
def _has_mi_labels(raw):  # 定义辅助函数，检查当前的脑电记录（mne的Raw对象）中是否含有指定的运动想象事件码
    ann_desc = set(raw.annotations.description) 
    return all(d in ann_desc for d in ['769', '770', '771', '772']) 


def preprocess_eeg(raw, l_freq=8, h_freq=30, resample_rate=128, tmin=-1, tmax=4):  # 定义EEG原始信号的预处理函数
  
    raw.pick_types(eeg=True, stim=False, exclude=[]) 
    raw.filter(l_freq, h_freq, fir_design='firwin') 
    raw.resample(resample_rate) 

    events, event_dict = mne.events_from_annotations(raw)  
    target_desc = ['769', '770', '771', '772']  
    event_id = {desc: event_dict[desc] for desc in target_desc if desc in event_dict}  
    label_remap = {event_dict[desc]: i for i, desc in enumerate(target_desc) if desc in event_dict}  

    reject_criteria = dict(eeg=100e-6)

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,  
                        baseline=(None, 0), reject=reject_criteria, preload=True) 
    

    epochs.crop(tmin=0, tmax=4)
    epochs.set_eeg_reference(ref_channels='average')  

    data = epochs.get_data()  
    labels = np.array([label_remap[e] for e in epochs.events[:, -1]], dtype=np.int32)  
    return data, labels  


def subject_normalize(data):  # 定义在受试者样本层面的 Z-Score标准化操作函数
    mean = data.mean(axis=(0, 2), keepdims=True)  
    std = data.std(axis=(0, 2), keepdims=True)  
    return (data - mean) / (std + 1e-8) 


def load_bci2a_subject(subject_id, data_dir):
    """加载单个受试者的 T+E 两个 session 并合并.

    优先使用 T session (训练 session), 若 E session 也有类别标签则合并.
    合并后统一做逐通道 z-score 归一化.
    """
    session_files = [
        (f"{data_dir}/A{subject_id:02d}T.gdf", 'T'),
        (f"{data_dir}/A{subject_id:02d}E.gdf", 'E'),
    ]

    all_data = []
    all_labels = []
    used = []

    for fpath, stype in session_files:
        raw = read_raw_gdf(fpath, preload=True,
                           stim_channel="auto",
                           exclude=["EOG-left", "EOG-central", "EOG-right"],
                           verbose='ERROR')
        if _has_mi_labels(raw):
            data, labels = preprocess_eeg(raw)
            all_data.append(data)
            all_labels.append(labels)
            used.append(stype)
        else:
            print(f"  注意: {fpath} ({stype} session) 缺少类别标注, 已跳过")

    if not all_data:
        raise RuntimeError(f"受试者 A{subject_id:02d} 没有任何包含 MI 标签的文件")

    data_all = np.concatenate(all_data, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    data_all = subject_normalize(data_all)
    print(f"  A{subject_id:02d}: session={used}, trials={len(data_all)}, "
          f"per_class={[(labels_all==c).sum() for c in range(4)]}")
    
   
    if 'E' not in used:
        # 手动切分 T session 数据为训练和测试
        half_idx = len(data_all) // 2
        train_ds = EEGSubjectDataset(data_all[:half_idx], labels_all[:half_idx])
        test_ds = EEGSubjectDataset(data_all[half_idx:], labels_all[half_idx:])
        return train_ds, test_ds
    else:
        # 如果 E session 有标签，进行分离
        train_len = len(all_data[0])
        train_ds = EEGSubjectDataset(data_all[:train_len], labels_all[:train_len])
        test_ds = EEGSubjectDataset(data_all[train_len:], labels_all[train_len:])
        return train_ds, test_ds
