"""
BCI2a LOSO 元训练 — 严格复现原论文 TCPL 模型.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from models.tcpl import TCPL
from data.dataset import load_bci2a_subject, sample_task, sample_test_task
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompt_l2_sq(prompts):
    return (prompts ** 2).sum()


def collate_to_device(batch, device):
    data = torch.stack([item[0] for item in batch]).to(device)
    labels = torch.tensor([item[1] for item in batch]).to(device)
    return data, labels


def evaluate(model, train_ds, test_ds, shots, n_classes, device, n_sets=5):
    """从 T session (train_ds) 采样 n_sets 个不同 support set，对 E session (test_ds) 进行预测."""
    model.eval()
    accs = []
    per_class_all = []

    for _ in range(n_sets):
        # 支撑集从 T session (训练集) 中采样
        support_ds, _ = sample_task(train_ds, shots, q_per_class=0)
        # 查询集使用整个 E session (测试集)
        query_ds = test_ds

        support_data, support_labels = collate_to_device(
            [(support_ds[i][0], support_ds[i][1]) for i in range(len(support_ds))],
            device)
        query_data, query_labels = collate_to_device(
            [(query_ds[i][0], query_ds[i][1]) for i in range(len(query_ds))],
            device)

        preds, _ = model.predict_by_support(support_data, query_data, support_labels)

        correct = (preds == query_labels)
        acc = correct.float().mean().item()
        accs.append(acc)

        pc = {}
        for c in range(n_classes):
            mask = (query_labels == c)
            if mask.sum() > 0:
                pc[c] = correct[mask].float().mean().item()
            else:
                pc[c] = 0.0
        per_class_all.append(pc)

    # 平均 per-class
    avg_pc = {}
    for c in range(n_classes):
        avg_pc[c] = np.mean([pc[c] for pc in per_class_all])

    return np.mean(accs), avg_pc


def meta_train_loso(config, seed=None):
    """LOSO 留一交叉验证 — 严格按照论文设计."""
    if seed is not None:
        config.seed = seed
    set_seed(config.seed)

    all_subject_ids = list(range(1, 10))
    accs = []
    per_class_accs = []

    # 保存目录
    save_dir = os.path.join(os.path.dirname(__file__), "results",
                            datetime.now().strftime("%Y%m%d_%H%M%S") + "_strict")
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果保存目录: {save_dir}")
    print(f"随机种子: {config.seed}")
    print(f"设备: {config.device}")

    # 加载全部受试者数据
    print("\n加载数据...")
    datasets = {}
    for sid in all_subject_ids:
        train_ds, test_ds = load_bci2a_subject(sid, config.data_dir)
        datasets[sid] = {'train': train_ds, 'test': test_ds}
        if sid == 1:
            sample_data, _ = train_ds[0]
            config.n_channels = sample_data.shape[0]
            config.n_times = sample_data.shape[1]
            print(f"  自动设置: n_channels={config.n_channels}, n_times={config.n_times}")

    n_classes = config.n_classes
    
    # 论文设定 batch_size_episodes = 16
    config.batch_size_episodes = getattr(config, 'batch_size_episodes', 16)
    config.tasks_per_subject = getattr(config, 'tasks_per_subject', 5)

    # LOSO 交叉验证
    for test_id in all_subject_ids:
        print(f"\n{'='*60}")

        train_ids = [sid for sid in all_subject_ids if sid != test_id]
        val_id = train_ids[-1]
        actual_train_ids = train_ids[:-1]
        n_train_actual = len(actual_train_ids)

        effective_batch = n_train_actual * config.tasks_per_subject
        actual_batch_size = min(config.batch_size_episodes, effective_batch)

        print(f"Fold {test_id}: Test=A{test_id:02d}, Val=A{val_id:02d}, "
              f"Train={[f'A{s:02d}' for s in actual_train_ids]}, "
              f"effective_batch={effective_batch}, actual_batch={actual_batch_size}")

        model = TCPL(config).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs)

        best_val_acc = 0.0
        best_epoch = 0
        best_model_state = None
        fold_log = []

        for epoch in range(config.epochs):
            model.train()
            epoch_losses = []
            episode_count = 0

            for _ in range(config.tasks_per_subject):
                shuffled_ids = actual_train_ids.copy()
                random.shuffle(shuffled_ids)
                for sid in shuffled_ids:
                    support_ds, query_ds = sample_task(
                        datasets[sid]['train'], config.shots, config.query_samples)
                    if len(support_ds) == 0 or len(query_ds) == 0:
                        continue

                    support_data, support_labels = collate_to_device(
                        [(support_ds[i][0], support_ds[i][1])
                         for i in range(len(support_ds))],
                        config.device)
                    query_data, query_labels = collate_to_device(
                        [(query_ds[i][0], query_ds[i][1])
                         for i in range(len(query_ds))],
                        config.device)

                    logits_tcpl, prompts = model.meta_train_step(
                        support_data, query_data, support_labels)

                    # TCPL 分类损失
                    ce_loss = nn.CrossEntropyLoss()(logits_tcpl, query_labels)

                    # 正则化
                    reg_loss = prompt_l2_sq(prompts) * config.lambda_reg

                    # 总损失只有 CE + 正则
                    loss = ce_loss + reg_loss
                    loss = loss / actual_batch_size
                    loss.backward()

                    epoch_losses.append(loss.item() * actual_batch_size)
                    episode_count += 1

                    if episode_count % actual_batch_size == 0:
                        optimizer.step()
                        optimizer.zero_grad()

            # 处理剩余梯度
            if episode_count % actual_batch_size != 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # 验证
            val_acc, val_pc = evaluate(model, datasets[val_id]['train'], datasets[val_id]['test'],
                                       config.shots, n_classes, config.device,
                                       n_sets=getattr(config, 'eval_support_sets', 5))
            fold_log.append((epoch + 1, avg_loss, val_acc, val_pc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone()
                                   for k, v in model.state_dict().items()}

            per_class_str = "  ".join(
                [f"c{c}:{val_pc.get(c,0):.3f}" for c in range(n_classes)])
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Val: {val_acc:.4f} | Best: {best_val_acc:.4f} | [{per_class_str}]")

        # 加载最佳模型，在测试受试者上评估
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最终测试阶段：固定多次采样 (n_sets=5)
        test_acc, test_pc = evaluate(model, datasets[test_id]['train'], datasets[test_id]['test'],
                                     config.shots, n_classes, config.device,
                                     n_sets=5)

        # 保存
        if best_model_state is not None:
            torch.save(best_model_state,
                       os.path.join(save_dir, f"best_model_A{test_id:02d}.pt"))
        with open(os.path.join(save_dir, f"val_log_A{test_id:02d}.csv"), 'w') as f:
            f.write("epoch,loss,val_acc," + ",".join(f"c{c}" for c in range(n_classes)) + "\n")
            for ep, loss, acc, pc in fold_log:
                f.write(f"{ep},{loss:.6f},{acc:.6f},"
                        + ",".join(f"{pc.get(c,0):.6f}" for c in range(n_classes)) + "\n")

        accs.append(test_acc)
        per_class_accs.append(test_pc)
        print(f"  A{test_id:02d} 测试准确率: {test_acc:.4f} (best epoch {best_epoch})")

    # 汇总
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\n{'='*60}")
    print(f"LOSO 结果: {mean_acc:.4f} ± {std_acc:.4f}")
    for i, test_id in enumerate(all_subject_ids):
        pc = per_class_accs[i]
        pc_str = "  ".join([f"c{c}:{pc.get(c,0):.3f}" for c in range(n_classes)])
        print(f"  A{test_id:02d}: {accs[i]:.4f}  [{pc_str}]")

    summary_path = os.path.join(save_dir, "summary.csv")
    with open(summary_path, 'w') as f:
        f.write("test_subject,test_acc," + ",".join(f"c{c}" for c in range(n_classes)) + "\n")
        for i, test_id in enumerate(all_subject_ids):
            pc = per_class_accs[i]
            f.write(f"A{test_id:02d},{accs[i]:.6f},"
                    + ",".join(f"{pc.get(c,0):.6f}" for c in range(n_classes)) + "\n")
        f.write(f"mean,{mean_acc:.6f}\n")
        f.write(f"std,{std_acc:.6f}\n")
    print(f"汇总: {summary_path}")

    return accs, save_dir


if __name__ == "__main__":
    cfg = Config()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 论文设定：所有实验重复 5 个种子，报告均值±标准差
    seeds = [42, 43, 44, 45, 46]
    all_seed_accs = []
    
    for seed in seeds:
        print(f"\n\n{'='*80}\n开始执行 Seed: {seed}\n{'='*80}")
        cfg.seed = seed
        accs, save_dir = meta_train_loso(cfg, seed=seed)
        all_seed_accs.append(accs)
        
    all_seed_accs = np.array(all_seed_accs)  # 形如 (5_seeds, 9_subjects)
    
    print(f"\n\n{'*'*80}")
    print(f"5 个种子的最终结果总汇:")
    
    # 计算每个受试者在5个种子上的平均和标准差
    for i, sid in enumerate(range(1, 10)):
        sub_mean = all_seed_accs[:, i].mean()
        sub_std = all_seed_accs[:, i].std()
        print(f"  A{sid:02d}: {sub_mean:.4f} ± {sub_std:.4f}")
        
    # 计算各种子整体均值然后再求均值和标准差
    seed_means = all_seed_accs.mean(axis=1) # 5个种子的均值
    grand_mean = seed_means.mean()
    grand_std = seed_means.std()
    
    print(f"\n整体最终 LOSO 结果: {grand_mean:.4f} ± {grand_std:.4f}")
    print(f"{'*'*80}\n")
