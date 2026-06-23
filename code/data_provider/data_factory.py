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

    seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, mask_list = zip(*batch)
    batch_x = torch.tensor(np.stack(seq_x_list, axis=0), dtype=torch.float32)
    batch_y = torch.tensor(np.stack(seq_y_list, axis=0), dtype=torch.float32)
    batch_x_mark = torch.tensor(np.stack(seq_x_mark_list, axis=0), dtype=torch.float32)
    batch_y_mark = torch.tensor(np.stack(seq_y_mark_list, axis=0), dtype=torch.float32)
    mask = torch.tensor(np.stack(mask_list, axis=0), dtype=torch.float32)
    new_batch_size = batch_x.shape[0] * batch_x.shape[1]
    batch_x = batch_x.reshape(new_batch_size, batch_x.shape[2], batch_x.shape[3])
    batch_y = batch_y.reshape(new_batch_size, batch_y.shape[2], batch_y.shape[3])
    batch_x_mark = batch_x_mark.reshape(new_batch_size, batch_x_mark.shape[2], batch_x_mark.shape[3])
    batch_y_mark = batch_y_mark.reshape(new_batch_size, batch_y_mark.shape[2], batch_y_mark.shape[3])
    mask = mask.reshape(new_batch_size, mask.shape[2], mask.shape[3])
    return (
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        mask
    )

def create_worker_init_fn(base_seed):

    def worker_init_fn(worker_id):
        seed = (base_seed + worker_id) % (2**32)  # 防止整数溢出
        print(f"[Worker {worker_id}] Setting seed={seed}")  # 调试
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    return worker_init_fn

def safe_load_npy(path):
        data = np.load(path)
        if isinstance(data, np.memmap):
            data = data.copy()
        return data
def data_provider(args, flag, train_generator=None,base_seed=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = True if (flag == 'train' or flag == 'TRAIN') else False
    drop_last = True if (flag == 'train' or flag == 'TRAIN') else False
    freq = args.freq
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
    generator = None
    pin_memory = getattr(args, 'use_gpu', False)

    if is_train:
        if train_generator is None:
            raise ValueError("train_generator is required for training (imputation)")
        generator = train_generator
    else:
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
        persistent_workers=False,
        generator=generator,
        worker_init_fn=worker_init_fn_func,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    return data_set, data_loader