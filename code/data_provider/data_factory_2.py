# data_provider.py

from data_provider.data_loader import Dataset_Custom_l,Dataset_Custom_0,Dataset_Custom_s0
from torch.utils.data import DataLoader
from torch import Generator
import numpy as np
import torch
import os
import random

data_dict = {
    'customl': Dataset_Custom_l,
    'custom0': Dataset_Custom_0,
    'customs0':Dataset_Custom_s0
}
import torch
import numpy as np
from typing import List, Tuple, Any


def custom_collate_fn_final(batch: List[Any], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    自定义 collate_fn，适配 [B, N, T, C] 形状，执行 B 和 N 维度的合并。
    """

    seq_x_list, seq_x_e_list, seq_y_list, s_list, e_list, datee_list, o_mask_list = zip(*batch)

    # 结果形状: [B, ...]
    # 注意：对于 s 和 e (locations/embeddings)，如果它们是节点嵌入或图结构，通常只需要取 batch 中的第一个元素，因为它们在所有样本中可能相同，但此处我们先按标准堆叠。

    batch_x = torch.tensor(np.stack(seq_x_list, axis=0), dtype=torch.float32)  # [B, C, N, T]
    batch_x_e = torch.tensor(np.stack(seq_x_e_list, axis=0), dtype=torch.float32)  # [B, C', N, T]
    batch_y = torch.tensor(np.stack(seq_y_list, axis=0), dtype=torch.float32)  # [B, C, N, T]
    batch_s = torch.tensor(np.stack(s_list, axis=0), dtype=torch.float32)  # [B, ...]
    batch_e = torch.tensor(np.stack(e_list, axis=0), dtype=torch.float32)  # [B, ...]
    batch_datee = torch.tensor(np.stack(datee_list, axis=0), dtype=torch.float32)  # [B, C'', T, D]
    batch_o_mask = torch.tensor(np.stack(o_mask_list, axis=0), dtype=torch.float32)  # [B, M, N, T]

    # 2. 整形 (Reshaping): 合并 Batch (B) 和 站点 (N) 维度

    # 2. 整形 (Reshaping): 合并 Batch (B) 和 站点 (N) 维度
    # [B, N, T, C] -> [B*N, T, C]

    # 获取新的 Batch Size (B*N)
    new_batch_size = batch_x.shape[0] * batch_x.shape[1]

    # 合并 B*N 维度
    batch_x = batch_x.reshape(new_batch_size, batch_x.shape[2], batch_x.shape[3],batch_x.shape[4])
    batch_y = batch_y.reshape(new_batch_size, batch_y.shape[2], batch_y.shape[3],batch_y.shape[4])
    batch_x_e = batch_x_e.reshape(new_batch_size, batch_x_e.shape[2], batch_x_e.shape[3],batch_x_e.shape[4])
    batch_datee = batch_datee.reshape(new_batch_size, batch_datee.shape[2], batch_datee.shape[3])
    batch_o_mask= batch_o_mask.reshape(new_batch_size, batch_o_mask.shape[2], batch_o_mask.shape[3],batch_o_mask.shape[4])
    batch_s=batch_s[0]
    batch_e=batch_e[0]
    # 3. 传输到 GPU (一次性操作)
    return (
        batch_x,
        batch_x_e,
        batch_y,
        batch_s,
        batch_e,
        batch_datee,
        batch_o_mask
    )

def create_worker_init_fn(base_seed):
    """
    创建 worker 初始化函数，通过闭包捕获 base_seed
    避免依赖 worker_info.generator（PyTorch 多进程传递不稳定）
    """
    def worker_init_fn(worker_id):
        seed = (base_seed + worker_id) % (2**32)  # 防止整数溢出
        print(f"[Worker {worker_id}] Setting seed={seed}")  # 调试
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    return worker_init_fn

def safe_load_npy(path):
        """安全加载 .npy 文件，避免 memmap 持有文件句柄"""
        data = np.load(path)
        if isinstance(data, np.memmap):
            print(f"⚠️  memmap detected: {path}, copying to memory...")
            data = data.copy()  # 转为普通 ndarray
        return data
def data_provider(args, flag, train_generator=None,base_seed=None):


    """
    为 imputation 任务提供数据集和 DataLoader

    Args:
        args: 训练参数
        flag: 'train', 'val', 'test'
        train_generator: 训练用的 Generator（外部管理）

    Returns:
        data_set, data_loader
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = True if (flag == 'train' or flag == 'TRAIN') else False
    drop_last = True if (flag == 'train' or flag == 'TRAIN') else False
    freq = args.freq



    # 然后替换所有 np.load
    feature_dir = f"../{args.feature}"
    X_train_all = safe_load_npy(os.path.join(feature_dir, 'X_train_all.npy'))
    X_val_all = safe_load_npy(os.path.join(feature_dir, 'X_val_all.npy'))
    X_test_all = safe_load_npy(os.path.join(feature_dir, 'X_test_all.npy'))
    locations = safe_load_npy(os.path.join(feature_dir, 'locations.npy'))
    X_train_all_e = safe_load_npy(os.path.join(feature_dir, 'X_train_all_e.npy'))
    X_val_all_e = safe_load_npy(os.path.join(feature_dir, 'X_val_all_e.npy'))
    X_test_all_e = safe_load_npy(os.path.join(feature_dir, 'X_test_all_e.npy'))
    locations_e = safe_load_npy(os.path.join(feature_dir, 'locations_e.npy'))
    is_train=flag in ['train', 'TRAIN']
    # --- Generator 设置 ---
    generator = None
    pin_memory = getattr(args, 'use_gpu', False)

    if is_train:
        if train_generator is None:
            raise ValueError("train_generator is required for training (imputation)")
        generator = train_generator
    else:
        # val/test 使用固定 seed，保证结果一致
        seed_offset = 100 if flag == 'val' else 200
        generator = Generator()

        generator.manual_seed(args.seed + seed_offset)
        base_seed = generator.initial_seed()

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        set_size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns,
        X_train_all=X_train_all,
        X_val_all=X_val_all,
        X_test_all=X_test_all,
        locations=locations,
        X_train_all_e=X_train_all_e,
        X_val_all_e=X_val_all_e,
        X_test_all_e=X_test_all_e,
        locations_e=locations_e,
        )

    # --- DataLoader 参数 ---
    if flag in ['test', 'TEST']:
        batch_size =  args.batch_size
        num_workers=0
        pin_memory=False

    else:
        batch_size = args.batch_size
    if args.num_workers > 0:
        worker_init_fn_func = create_worker_init_fn(base_seed)
    else:
        worker_init_fn_func = None

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=lambda b: custom_collate_fn_final(b, args.device),
        persistent_workers=False,  # ✅ 避免每个 epoch 重建 worker
        generator=generator,  # ✅ 用于 shuffle 的随机性控制
        worker_init_fn=worker_init_fn_func,  # ✅ 用闭包，不再依赖 worker_info.generator
        prefetch_factor=2 if args.num_workers > 0 else None,  # ✅ 预取数据加速
    )

    return data_set, data_loader