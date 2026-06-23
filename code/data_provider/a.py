"""import os

import numpy as np
import torch



def safe_load_npy(path):
    
    data = np.load(path)
    if isinstance(data, np.memmap):
        print(f"⚠️  memmap detected: {path}, copying to memory...")
        data = data.copy()  # 转为普通 ndarray
    return data
feature="v10"
feature_dir = f"../../{feature}"
X_train_all = safe_load_npy(os.path.join(feature_dir, 'X_train_all.npy'))
X_train_all_e = safe_load_npy(os.path.join(feature_dir, 'X_train_all_e.npy'))
filename_mean = f"../scaler/{feature}_mean_n.npy"
filename_std = f"../scaler/{feature}_std_n.npy"
scaler_mean = np.mean(X_train_all, axis=1, keepdims=True)
np.save(filename_mean,scaler_mean)
print(f"标准化文件{filename_mean}已下载。")
scaler_std = np.std(X_train_all, axis=1, keepdims=True)
np.save(filename_std, scaler_std )
print(f"标准化文件{filename_std}已下载。")
print(scaler_mean.shape)
filename_mean_e = f"../scaler/{feature}_mean_e_n.npy"
filename_std_e = f"../scaler/{feature}_std_e_n.npy"
self_mean_e = np.mean(X_train_all_e, axis=(0, 1), keepdims=True)
np.save(filename_mean_e, self_mean_e)
print(f"标准化文件{filename_mean_e}已下载。")

self_std_e = np.std(X_train_all_e, axis=(0, 1), keepdims=True)
np.save(filename_std_e, self_std_e)
print(f"标准化文件{filename_std_e}已下载。")

#生成掩码
"""
"""
import os
import numpy as np
import torch


def safe_load_npy(path):
    data = np.load(path)
    if isinstance(data, np.memmap):
        print(f"⚠️  memmap detected: {path}, copying to memory...")
        data = data.copy()  # 转为普通 ndarray
    return data


features = ['msl', 'TMP', 'u10', 'v10']
mask_rates = [0.125, 0.25,0.5, 0.75]

for feature in features:
    for mask_rate in mask_rates:
        feature_dir = f"../../{feature}"
        X_test_all = safe_load_npy(os.path.join(feature_dir, 'X_train_all.npy'))
        print(f"Loaded {X_test_all.shape} from {feature_dir}")

        N, T, C = X_test_all.shape

        # 存储10次生成的mask
        test_masks_list = []

        for i in range(10):
            # 生成 [N, T, C] 的随机均匀分布张量
            mask = torch.rand((N, T, C))
            # 应用 mask_rate: <= mask_rate 的设为 0 (masked), > mask_rate 的设为 1 (unmasked)
            mask[mask <= mask_rate] = 0  # masked
            mask[mask > mask_rate] = 1  # unmasked
            test_masks_list.append(mask.numpy())

        # 堆叠成 (10, N, T, C)
        test_masks = np.stack(test_masks_list, axis=0)
        print(f"Generated stacked masks with shape: {test_masks.shape}")

        # 保存文件
        output_dir = f'../mask_rate/{mask_rate}'
        os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
        filename = f'{output_dir}/train_mask_{feature}_4.npy'
        np.save(filename, test_masks)
        print(f"Saved mask to {filename}")
        test=np.load(filename)
        mask_rate = np.mean(test == 0)
        print(test.shape,mask_rate)
"""
"""
import hashlib
import numpy as np

def get_npy_hash(file_path):
    
    arr = np.load(file_path)
    # 使用 .tobytes() 将数组数据转为字节流（最核心的内容）
    return hashlib.md5(arr.tobytes()).hexdigest()

# 示例：比较多个文件
files = [
    "../mask_rate/0.125/test_mask_u10.npy", "../mask_rate/0.25/test_mask_u10.npy",

    "../mask_rate/0.5/test_mask_u10.npy",

    "../mask_rate/0.75/test_mask_u10.npy",

]

print("文件名\t\t\t→ MD5 哈希值")
print("-" * 60)
for f in files:
    if os.path.exists(f):
        h = get_npy_hash(f)
        print(f"{f:<20} → {h}")
    else:
        print(f"{f} 不存在")
"""

import os

import numpy as np
import torch


def safe_load_npy(path):
    data = np.load(path)
    if isinstance(data, np.memmap):
        print(f"⚠️  memmap detected: {path}, copying to memory...")
        data = data.copy()  # 转为普通 ndarray
    return data


feature = "msl"
feature_dir = f"../../{feature}"
X_train_all = safe_load_npy(os.path.join(feature_dir, 'X_train_all.npy'))
X_train_all_e = safe_load_npy(os.path.join(feature_dir, 'X_train_all_e.npy'))
filename_mean = f"../scaler/{feature}_mean.npy"
filename_std = f"../scaler/{feature}_std.npy"
scaler_mean = np.mean(X_train_all, axis=(0,1) )
np.save(filename_mean, scaler_mean)
print(f"标准化文件{filename_mean}已下载。")
scaler_std = np.std(X_train_all, axis=(0,1))
np.save(filename_std, scaler_std)
print(f"标准化文件{filename_std}已下载。")
print(scaler_mean.shape)
print(scaler_mean)
filename_mean_e = f"../scaler/{feature}_mean_e.npy"
filename_std_e = f"../scaler/{feature}_std_e.npy"
self_mean_e = np.mean(X_train_all_e, axis=(0, 1))
np.save(filename_mean_e, self_mean_e)
print(f"标准化文件{filename_mean_e}已下载。")

self_std_e = np.std(X_train_all_e, axis=(0, 1))
np.save(filename_std_e, self_std_e)
print(f"标准化文件{filename_std_e}已下载。")
print(self_mean_e.shape)
print(self_mean_e)
"""
标准化文件../scaler/TMP_mean.npy已下载。
标准化文件../scaler/TMP_std.npy已下载。
(1,)
[22.12394616]
标准化文件../scaler/TMP_mean_e.npy已下载。
标准化文件../scaler/TMP_std_e.npy已下载。
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/msl_mean.npy已下载。
标准化文件../scaler/msl_std.npy已下载。
(1,)
[1012.88700502]
标准化文件../scaler/msl_mean_e.npy已下载。
标准化文件../scaler/msl_std_e.npy已下载。
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/u10_mean.npy已下载。
标准化文件../scaler/u10_std.npy已下载。
(1,)
[-0.55057355]
标准化文件../scaler/u10_mean_e.npy已下载。
标准化文件../scaler/u10_std_e.npy已下载。
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/u10_mean.npy已下载。
标准化文件../scaler/u10_std.npy已下载。
(1,)
[-0.55057355]
标准化文件../scaler/u10_mean_e.npy已下载。
标准化文件../scaler/u10_std_e.npy已下载。
(1,)
[-1.56597899]
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/v10_mean.npy已下载。
标准化文件../scaler/v10_std.npy已下载。
(1,)
[-0.08910787]
标准化文件../scaler/v10_mean_e.npy已下载。
标准化文件../scaler/v10_std_e.npy已下载。
(1,)
[-0.61264655]
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/TMP_mean.npy已下载。
标准化文件../scaler/TMP_std.npy已下载。
(1,)
[22.12394616]
标准化文件../scaler/TMP_mean_e.npy已下载。
标准化文件../scaler/TMP_std_e.npy已下载。
(1,)
[22.48280072]
(myenv3.9) ljw@visint51:/data/ljw/tmp_Gdong/mpnn/data_provider$ python a.py 
标准化文件../scaler/msl_mean.npy已下载。
标准化文件../scaler/msl_std.npy已下载。
(1,)
[1012.88700502]
标准化文件../scaler/msl_mean_e.npy已下载。
标准化文件../scaler/msl_std_e.npy已下载。
(1,)
[1012.63400667]
"""