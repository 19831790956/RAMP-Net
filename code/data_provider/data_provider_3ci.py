import os
import numpy as np
import pandas as pd
from utils.metrics import metric
import glob
import re
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from torch import nn
warnings.filterwarnings('ignore')
#标准化是采用ERA和STA分别全局标准化，参照MPNN论文 MULTI-MODAL GRAPH NEURAL NETWORKS FOR  LOCALIZED OFF-GRID WEATHER FORECASTING
""" 
        df_stamp['DATE'] = pd.to_datetime(df_stamp['DATE'])  # 转换为datetime格式

        # 提取时间特征
        df_stamp['YEAR'] = df_stamp['DATE'].dt.year  # 年份（如 2023）
        df_stamp['MONTH'] = df_stamp['DATE'].dt.month  # 月份（1-12）
        df_stamp['WEEKDAY'] = df_stamp['DATE'].dt.weekday
        df_stamp['HOUR'] = df_stamp['DATE'].dt.hour
        date_co = df_stamp[['YEAR', 'MONTH', 'WEEKDAY', 'HOUR']].values       """
import hashlib
def get_array_hash(array):
    # 将 numpy array 转为 bytes 并计算 md5 hash
    return hashlib.md5(array.tobytes()).hexdigest()

def fill_1d_with_spline_vectorized(X, mask):# 1维
    # 确保输入为numpy数组
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    # 获取数据维度
    T= X.shape[0]

    # 初始化填充后的数组
    filled_data = X.copy()

    # 创建时间索引数组
    time_indices = np.arange(T)


    # 提取当前时间序列和掩码
    ts = filled_data
    ts_mask = mask

    # 获取有效数据点
    valid_indices = time_indices[ts_mask]
    valid_values = ts[ts_mask]

    # 获取缺失数据点的索引
    missing_indices = time_indices[~ts_mask]


    # 三次样条插值
    interp_func = interp1d(
        valid_indices, valid_values,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )

    # 填充缺失值
    filled_data[missing_indices] = interp_func(missing_indices)

    return filled_data

def fill_1d_with_linear(X, mask):#1维
    # 线性插值仅针对第5列（索引4），mask 取第1列（索引0）
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    T = X.shape[0]
    filled_data = X.copy()
    time_indices = np.arange(T)

    ts = filled_data
    ts_mask = mask

    valid_indices = time_indices[ts_mask]
    missing_indices = time_indices[~ts_mask]

    if valid_indices.size == 0:
        # 无有效点，返回全 0（标准化域），由上层再做 nan_to_num/clip
        filled_data[:] = 0.0
        return filled_data

    if valid_indices.size == 1:
        # 仅 1 个点，全部用该点值
        filled_data[:] = ts[ts_mask][0]
        return filled_data

    interp_func = interp1d(
        valid_indices, ts[ts_mask], kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    filled_data[missing_indices] = interp_func(missing_indices)
    return filled_data
def safe_fill_4d_with_fallback(X_norm, mask, clip_value=5.0, min_points_cubic=4, min_points_linear=2):
        """
        稳健 4D 插值（整段）：按有效点阈值对每个 (n,c) 选择 cubic/linear/常数/零，并裁剪清洗。
        - X_norm: [N, T, C]
        - mask:   [N, T, C]
        可用于任意窗口长度 T（整段或 24 窗口）。
        """
        X = np.asarray(X_norm, dtype=float)
        mask = np.asarray(mask)
        if mask.dtype != bool:
            mask_bool = mask.astype(bool)
        else:
            mask_bool = mask
        N, T, C = X.shape
        out = X.copy()
        time_indices = np.arange(T)
        for n in range(N):
            for c in range(C):
                valid = mask_bool[n, :, c]
                valid_count = int(valid.sum())
                if valid_count >= min_points_cubic:
                    out[n, :, c]=fill_1d_with_spline_vectorized(X[n, :, c], valid)
                elif valid_count >= min_points_linear:
                    out[n, :, c]=fill_1d_with_linear(X[n, :, c], valid)
                elif valid_count == 1:
                    idx = np.where(valid)[0][0]
                    out[n, :, c] = X[n, idx, c]
                else:
                    out[n, :, c] = 0.0
                # 通道级裁剪与清洗
                out[n, :, c] = np.clip(out[n, :, c], -clip_value, clip_value)
                out[n, :, c] = np.nan_to_num(out[n, :, c], nan=0.0, posinf=clip_value, neginf=-clip_value)
        return out


def fill_2d_with_spline_vectorized(X, mask):
    # 确保输入为numpy数组
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    # 获取数据维度
    T, C = X.shape

    # 初始化填充后的数组
    filled_data = X.copy()

    # 创建时间索引数组
    time_indices = np.arange(T)


    # 提取当前时间序列和掩码
    ts = filled_data[:, 4]
    ts_mask = mask[:, 0]

    # 获取有效数据点
    valid_indices = time_indices[ts_mask]
    valid_values = ts[ts_mask]

    # 获取缺失数据点的索引
    missing_indices = time_indices[~ts_mask]


    # 三次样条插值
    interp_func = interp1d(
        valid_indices, valid_values,
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )

    # 填充缺失值
    filled_data[missing_indices, 4] = interp_func(missing_indices)

    return filled_data

def fill_2d_with_linear(X, mask):
    # 线性插值仅针对第5列（索引4），mask 取第1列（索引0）
    X = np.asarray(X, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    T, C = X.shape
    filled_data = X.copy()
    time_indices = np.arange(T)

    ts = filled_data[:, 4]
    ts_mask = mask[:, 0]

    valid_indices = time_indices[ts_mask]
    missing_indices = time_indices[~ts_mask]

    if valid_indices.size == 0:
        # 无有效点，返回全 0（标准化域），由上层再做 nan_to_num/clip
        filled_data[:, 4] = 0.0
        return filled_data

    if valid_indices.size == 1:
        # 仅 1 个点，全部用该点值
        filled_data[:, 4] = ts[ts_mask][0]
        return filled_data

    interp_func = interp1d(
        valid_indices, ts[ts_mask], kind='linear', bounds_error=False, fill_value="extrapolate"
    )
    filled_data[missing_indices, 4] = interp_func(missing_indices)
    return filled_data

def safe_fill_2d_with_fallback(X_norm, mask, clip_value=5.0, min_points_cubic=4, min_points_linear=2):
    """
    稳健填充：
    - 有效点 >=4: cubic 样条
    - 有效点 2~3: 线性插值
    - 有效点 1: 用该点常数填充
    - 有效点 0: 用 0.0（标准化域）
    之后进行裁剪与非有限值清洗。
    仅作用于第5列（索引4）；mask 使用第1列（索引0）。
    """
    X_norm = np.asarray(X_norm, dtype=float)
    mask = np.asarray(mask)

    # 计算有效点数
    if mask.dtype != bool:
        mask_bool = mask[:, 0].astype(bool)
    else:
        mask_bool = mask[:, 0]

    valid_count = int(mask_bool.sum())

    if valid_count >= min_points_cubic:
        filled = fill_2d_with_spline_vectorized(X_norm, mask_bool[:, None])
    elif valid_count >= min_points_linear:
        filled = fill_2d_with_linear(X_norm, mask_bool[:, None])
    elif valid_count == 1:
        filled = X_norm.copy()
        # 用唯一有效值常数填充
        idx = np.where(mask_bool)[0][0]
        filled[:, 4] = filled[idx, 4]
    else:
        filled = X_norm.copy()
        filled[:, 4] = 0.0  # 标准化域用 0 作为全局均值占位

    # 数值裁剪与清洗（仅对第5列）
    filled[:, 4] = np.clip(filled[:, 4], -clip_value, clip_value)
    filled[:, 4] = np.nan_to_num(filled[:, 4], nan=0.0, posinf=clip_value, neginf=-clip_value)
    return filled

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S',
                 target='OT',  timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 24 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate=args.mask_rate

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.axis=(0,1)
        self.keepdims=True
        self.mask=None
        self.scaler_mean=None
        self.scaler_std=None

        self.root_path = root_path
        self.data_path = data_path
        # 直接使用传入的参数
        self.X_train_all = X_train_all
        self.X_val_all = X_val_all
        self.X_test_all = X_test_all
        self.locations = locations
        self.X_train_all_e = X_train_all_e
        self.X_val_all_e = X_val_all_e
        self.X_test_all_e = X_test_all_e
        self.locations_e = locations_e
        self.__read_data__()

    def __read_data__(self):

        print('succeddful')
        test_true = self.X_test_all
        N, T, C = self.X_train_all.shape[0], self.X_train_all.shape[1], self.X_train_all.shape[2]
        _, T1, C1 = self.X_val_all.shape[0], self.X_val_all.shape[1], self.X_val_all.shape[2]
        _, T2, C2 = self.X_test_all.shape[0], self.X_test_all.shape[1], self.X_test_all.shape[2]
        print(self.X_train_all[0,0:2,:],self.X_val_all[0,0:3,:],self.X_test_all[0,0:4,:])
        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            self.scaler_mean=np.load(filename_mean)
            self.scaler_std=np.load(filename_std)
            X_train_all_norm = (self.X_train_all - self.scaler_mean) / self.scaler_std
            X_val_all_norm = (self.X_val_all - self.scaler_mean) / self.scaler_std
            X_test_all_norm = (self.X_test_all - self.scaler_mean) / self.scaler_std
            # 保存归一化副本，供窗口级插值使用
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
            '''X_train_all_fill=fill_4d_with_spline_vectorized(X_train_all_norm,train_mask)'''

            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_e_norm = (self.X_train_all_e - self_mean_e) / self_std_e
            X_val_all_e_norm = (self.X_val_all_e - self_mean_e) / self_std_e
            X_test_all_e_norm = (self.X_test_all_e - self_mean_e) / self_std_e

            #print(f'格点最终形状:{self_mean_e.shape},{self_std_e.shape},{X_train_all_e_norm.shape},{X_val_all_e_norm.shape},{X_test_all_e_norm.shape}')
        #print(f'数据集长度{T}{T1}{T2}')
        # 生成可重现的随机矩阵（使用现代化方法）
        #print('save success')
        #print("random_matrix hash:", get_array_hash(random_matrix_test))
        rng = np.random.default_rng(43)#42msl 43tmp 44u 45v
        #print(f'Random matrix shape: {random_matrix.shape}')
        #print(f'Random matrix unique values: {np.unique(random_matrix)}')
        if self.set_type == 0:
            filename=f'./mask_rate/{self.args.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                train_masks = np.load(filename)
            train_masks = np.array(train_masks)  # [10,N,T,1 or C]
        elif self.set_type == 1:
            filename=f'./mask_rate/{self.args.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                val_masks = np.load(filename)
            val_masks = np.array(val_masks)
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                test_masks = np.load(filename)
                print( test_masks.shape)
            test_masks = np.array(test_masks)
        df_date = pd.read_csv('./data_provider/split_date.csv')
        df_stamp = df_date[['DATE']]
        data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        date_co = data_stamp[:,[0,2,3]]
        print(" 0.5 标准")
        print(date_co.shape)
        if self.set_type == 0:
            self.data_x_e = X_train_all_e_norm
            self.data_y = X_train_all_norm
            self.date_=date_co[0:12000,:]
            self.mask=train_masks
        elif self.set_type == 1:
            self.data_x_e = X_val_all_e_norm
            self.data_y = X_val_all_norm
            self.date_ = date_co[12000:13680, :]
            self.mask=val_masks
        else:
            self.data_x_e = X_test_all_e_norm
            self.data_y = test_true
            self.date_ = date_co[13680:, :]
            self.mask=test_masks
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y,
                                                                                  self.args)
            #print('ssdsdooedmoc,dlc,l')
        #print("数据增强 hash:", get_array_hash(self.data_x), get_array_hash(self.data_y))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 窗口级 4D 稳健插值（mask 同 X 形状）
        o_mask = self.mask[:,:, s_begin:s_end, :]
        seq_x_e = self.data_x_e[:, s_begin:s_end, :]
        # 选择归一化基底
        if self.set_type == 0:
            base_norm = self.X_train_all_norm
        elif self.set_type == 1:
            base_norm = self.X_val_all_norm
        else:
            base_norm = self.X_test_all_norm
        window_norm = base_norm[:, s_begin:s_end, :]  # [N,24,C]
        filled_list = []
        min_points_cubic = getattr(self.args, 'min_points_cubic', 4)
        min_points_linear = getattr(self.args, 'min_points_linear', 2)
        for i in range(o_mask.shape[0]):
            filled = safe_fill_4d_with_fallback(
                window_norm, o_mask[i, :, :, :],
                clip_value=5.0,
                min_points_cubic=min_points_cubic,
                min_points_linear=min_points_linear
            )
            filled_list.append(filled)
        seq_x = np.stack(filled_list, axis=0)  # [10,N,24,C]
        seq_y = self.data_y[:, s_begin:s_end, :]
        s = self.locations
        e = self.locations_e
        datee=self.date_[s_begin:s_end, :]
        #print('日期形状')
        #print(datee.shape)
        #print(datee)
        # seq_x_mark = self.data_stamp[s_begin:s_end,:]
        # print(f'datemask的维度{seq_x_mark.shape}')

        return seq_x, seq_x_e, seq_y, s, e,datee,o_mask

    def __len__(self):
        return len(self.data_y[0]) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean



class Dataset_Custom_s(Dataset):
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S', target='OT', timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 24 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate = args.mask_rate
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.node_num=args.node
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_mean=None
        self.scaler_std=None
        self.__read_data__()

    def __read_data__(self):
        '''
        将maskN T C 每次获取1和 TC 参与以下运算
        '''
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        num_train, num_val, num_test = 12000, 1680, 3840
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        X_train_all=df_data[border1s[0]:border2s[0]].values
        X_val_all = df_data[border1s[1]:border2s[1]].values
        X_test_all = df_data[border1s[2]:border2s[2]].values
        self.trues = df_data[border1s[2]:border2s[2]].values
        #print('训练验证测试')
        #print(X_train_all.shape,X_val_all.shape,X_test_all.shape)
        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            self.scaler_mean = np.load(filename_mean)
            self.scaler_std = np.load(filename_std)
            print(self.scaler_mean,self.scaler_std)
            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_norm=X_train_all.copy()
            X_val_all_norm=X_val_all.copy()
            X_test_all_norm=X_test_all.copy()
            X_train_all_norm[:,4]=(X_train_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_train_all_norm[:, 0:4] = (X_train_all_norm[:, 0:4] - self_mean_e) / self_std_e
            X_val_all_norm[:,4]=(X_val_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_val_all_norm[:, 0:4] = (X_val_all_norm[:, 0:4] - self_mean_e) / self_std_e
            X_test_all_norm[:,4]=(X_test_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_test_all_norm[:, 0:4] = (X_test_all_norm[:, 0:4] - self_mean_e) / self_std_e
            # 持久化归一化序列，供 __getitem__ 窗口级插值使用
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
        features=1
        #print(f'Random matrix shape: {random_matrix.shape}')
        #print(f'Random matrix unique values: {np.unique(random_matrix)}')
        if self.set_type == 0:
            filename=f'./mask_rate/{self.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            train_masks = np.load(filename)
            print(f"加载文件名{filename}")
            print("random_matrix hash:", get_array_hash(train_masks[:,self.node_num,:,:]))
            train_masks=train_masks[:, self.node_num, :, :]
            train_masks = np.array(train_masks)  # shape: [10, T, C]
        elif self.set_type == 1:
            filename = f'./mask_rate/{self.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            val_masks = np.load(filename)
            print(f"加载文件名{filename}")
            print("random_matrix hash:", get_array_hash(val_masks[:, self.node_num, :, :]))
            val_masks = val_masks[:, self.node_num, :, :]
            val_masks = np.array(val_masks)  # shape: [10, T, C]
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
            print(f"加载文件名{filename}")

            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                test_masks = np.load(filename)
                test_masks=test_masks[:, self.node_num, :, :]
                print( test_masks.shape)#10 tc
            test_masks = np.array(test_masks)

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['DATE']][border1:border2]

        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['DATE'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_y = X_train_all_norm
            self.mask=train_masks
        elif self.set_type == 1:
            self.data_y = X_val_all_norm
            self.mask=val_masks
        else:
            self.data_y = X_test_all
            self.mask=test_masks

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 窗口级插值：对 10 个掩码在 24 步窗口内动态填补
        o_mask = self.mask[:, s_begin:s_end, :] # 10 24 1
        # 选择归一化基底序列
        if self.set_type == 0:
            base_norm = self.X_train_all_norm
        elif self.set_type == 1:
            base_norm = self.X_val_all_norm
        else:
            base_norm = self.X_test_all_norm
        window_norm = base_norm[s_begin:s_end, :]  # [24, C] #y
        filled_list = []
        min_points_cubic = getattr(self.args, 'min_points_cubic', 4)
        min_points_linear = getattr(self.args, 'min_points_linear', 2)
        for i in range(o_mask.shape[0]):
            filled = safe_fill_2d_with_fallback(
                window_norm, o_mask[i, :, :],
                clip_value=5.0,
                min_points_cubic=min_points_cubic,
                min_points_linear=min_points_linear
            )
            filled_list.append(filled)
        seq_x = np.stack(filled_list, axis=0)  # [10, 24, C]
        seq_y = self.data_y[s_begin:s_end,:]
        seq_x_mark = self.data_stamp[s_begin:s_end,:]
        seq_y_mark = self.data_stamp[s_begin:s_end,:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,o_mask

    def __len__(self):
        return len(self.data_y) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean



class Dataset_Custom_l(Dataset):
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S', target='OT', timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 1 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate = args.mask_rate
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.node_num=args.node
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_mean=None
        self.scaler_std=None
        self.__read_data__()

    def __read_data__(self):
        '''
        将maskN T C 每次获取1和 TC 参与以下运算
        '''
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        num_train, num_val, num_test = 12000, 1680, 3840
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, len(df_raw)]

        X_train_i=df_data[['f1','f2','f3','f4']][border1s[0]:border2s[0]].values
        X_train_o = df_data[['target']][border1s[0]:border2s[0]].values
        X_val_i=df_data[['f1','f2','f3','f4']][border1s[1]:border2s[1]].values
        X_val_o = df_data[['target']][border1s[1]:border2s[1]].values
        X_test_i=df_data[['f1','f2','f3','f4']][border1s[2]:border2s[2]].values
        X_test_o =df_data[['target']][border1s[2]:border2s[2]].values
        self.trues = df_data[['target']][border1s[2]:border2s[2]].values
        #print('训练验证测试')


        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            if all(os.path.exists(f) for f in [filename_mean, filename_std, filename_mean_e, filename_std_e]):
                print("✅ 所有标准化参数文件已存在：")
                self.scaler_mean = np.load(filename_mean)
                self.scaler_std = np.load(filename_std)
                scaler_e=np.load(filename_mean_e)
                std_e=np.load(filename_std_e)
                print("均值方差")
                print(self.scaler_mean,self.scaler_std,scaler_e,std_e)

                X_train_i = (X_train_i - scaler_e) / std_e
                X_train_o=(X_train_o - self.scaler_mean) / self.scaler_std
                X_val_i = (X_val_i - scaler_e) / std_e
                X_val_o = (X_val_o - self.scaler_mean) / self.scaler_std
                X_test_i = (X_test_i - scaler_e) / std_e
                X_test_o = X_test_o

            else:
                print("标准化文件不存在")


        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
        test_masks = np.load(filename)
        test_mask=test_masks[:,self.node_num,:,:]
        test_mask=test_mask.reshape(-1,1)

        X_test_i = np.tile(X_test_i[np.newaxis, :, :], (10, 1, 1)).reshape(-1, 4)
        X_test_o = np.tile(X_test_o[np.newaxis, :, :], (10, 1, 1)).reshape(-1, 1)

        print(X_train_i.shape, X_train_o.shape, X_val_i.shape, X_val_o.shape, X_test_i.shape, X_test_o.shape,test_mask.shape)
        if self.set_type == 0:
            self.data_x = X_train_i
            self.data_y = X_train_o
            self.mask = test_mask
        elif self.set_type == 1:
            self.data_x = X_val_i
            self.data_y =  X_val_o
            self.mask = test_mask
        else:
            self.data_x = X_test_i
            self.data_y = X_test_o
            self.mask=test_mask

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[s_begin:s_end,:]
        mask=self.mask[s_begin:s_end,:]
        return seq_x,seq_y,mask



    def __len__(self):
        return len(self.data_y) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean
class Dataset_Custom_0(Dataset):#填充零
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S',
                 target='OT',  timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 24 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate=args.mask_rate

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.axis=(0,1)
        self.keepdims=True
        self.mask=None
        self.scaler_mean=None
        self.scaler_std=None

        self.root_path = root_path
        self.data_path = data_path
        # 直接使用传入的参数
        self.X_train_all = X_train_all
        self.X_val_all = X_val_all
        self.X_test_all = X_test_all
        self.locations = locations
        self.X_train_all_e = X_train_all_e
        self.X_val_all_e = X_val_all_e
        self.X_test_all_e = X_test_all_e
        self.locations_e = locations_e
        self.__read_data__()

    def __read_data__(self):

        print('succeddful')
        test_true = self.X_test_all
        N, T, C = self.X_train_all.shape[0], self.X_train_all.shape[1], self.X_train_all.shape[2]
        _, T1, C1 = self.X_val_all.shape[0], self.X_val_all.shape[1], self.X_val_all.shape[2]
        _, T2, C2 = self.X_test_all.shape[0], self.X_test_all.shape[1], self.X_test_all.shape[2]
        print(self.X_train_all[0,0:2,:],self.X_val_all[0,0:3,:],self.X_test_all[0,0:4,:])
        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            self.scaler_mean=np.load(filename_mean)
            self.scaler_std=np.load(filename_std)
            X_train_all_norm = (self.X_train_all - self.scaler_mean) / self.scaler_std
            X_val_all_norm = (self.X_val_all - self.scaler_mean) / self.scaler_std
            X_test_all_norm = (self.X_test_all - self.scaler_mean) / self.scaler_std
            # 保存归一化副本，供窗口级插值使用
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
            '''X_train_all_fill=fill_4d_with_spline_vectorized(X_train_all_norm,train_mask)'''

            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_e_norm = (self.X_train_all_e - self_mean_e) / self_std_e
            X_val_all_e_norm = (self.X_val_all_e - self_mean_e) / self_std_e
            X_test_all_e_norm = (self.X_test_all_e - self_mean_e) / self_std_e

            #print(f'格点最终形状:{self_mean_e.shape},{self_std_e.shape},{X_train_all_e_norm.shape},{X_val_all_e_norm.shape},{X_test_all_e_norm.shape}')
        #print(f'数据集长度{T}{T1}{T2}')
        # 生成可重现的随机矩阵（使用现代化方法）
        #print('save success')
        #print("random_matrix hash:", get_array_hash(random_matrix_test))
        rng = np.random.default_rng(43)#42msl 43tmp 44u 45v
        #print(f'Random matrix shape: {random_matrix.shape}')
        #print(f'Random matrix unique values: {np.unique(random_matrix)}')
        if self.set_type == 0:
            filename=f'./mask_rate/{self.args.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                train_masks = np.load(filename)
            train_masks = np.array(train_masks)  # [10,N,T,1 or C]
        elif self.set_type == 1:
            filename=f'./mask_rate/{self.args.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                val_masks = np.load(filename)
            val_masks = np.array(val_masks)
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                test_masks = np.load(filename)
                print( test_masks.shape)
            test_masks = np.array(test_masks)
        df_date = pd.read_csv('./data_provider/split_date.csv')
        df_stamp = df_date[['DATE']]
        data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        date_co = data_stamp[:,[0,2,3]]
        print(" 0.5 标准")
        print(date_co.shape)
        if self.set_type == 0:
            self.data_x_e = X_train_all_e_norm
            self.data_y = X_train_all_norm
            self.date_=date_co[0:12000,:]
            self.mask=train_masks
        elif self.set_type == 1:
            self.data_x_e = X_val_all_e_norm
            self.data_y = X_val_all_norm
            self.date_ = date_co[12000:13680, :]
            self.mask=val_masks
        else:
            self.data_x_e = X_test_all_e_norm
            self.data_y = test_true
            self.date_ = date_co[13680:, :]
            self.mask=test_masks
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y,
                                                                                  self.args)
            #print('ssdsdooedmoc,dlc,l')
        #print("数据增强 hash:", get_array_hash(self.data_x), get_array_hash(self.data_y))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 窗口级 4D 稳健插值（mask 同 X 形状）
        o_mask = self.mask[:,:, s_begin:s_end, :]
        seq_x_e = self.data_x_e[:, s_begin:s_end, :]
        # 选择归一化基底
        if self.set_type == 0:
            base_norm = self.X_train_all_norm
        elif self.set_type == 1:
            base_norm = self.X_val_all_norm
        else:
            base_norm = self.X_test_all_norm
        window_norm = base_norm[:, s_begin:s_end, :]  # [N,24,C]
        filled_list=[]
        for i in range(o_mask.shape[0]):

            filled=window_norm*o_mask[i, :, :, :]

            filled_list.append(filled)
        seq_x = np.stack(filled_list, axis=0)  # [10,N,24,C]
        seq_y = self.data_y[:, s_begin:s_end, :]
        s = self.locations
        e = self.locations_e
        datee=self.date_[s_begin:s_end, :]
        #print('日期形状')
        #print(datee.shape)
        #print(datee)
        # seq_x_mark = self.data_stamp[s_begin:s_end,:]
        # print(f'datemask的维度{seq_x_mark.shape}')

        return seq_x, seq_x_e, seq_y, s, e,datee,o_mask

    def __len__(self):
        return len(self.data_y[0]) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean
class Dataset_Custom_self(Dataset):
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S',
                 target='OT',  timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 24 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate=args.mask_rate

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.axis=(0,1)
        self.keepdims=True
        self.mask=None
        self.scaler_mean=None
        self.scaler_std=None

        self.root_path = root_path
        self.data_path = data_path
        # 直接使用传入的参数
        self.X_train_all = X_train_all
        self.X_val_all = X_val_all
        self.X_test_all = X_test_all
        self.locations = locations
        self.X_train_all_e = X_train_all_e
        self.X_val_all_e = X_val_all_e
        self.X_test_all_e = X_test_all_e
        self.locations_e = locations_e
        self.__read_data__()

    def __read_data__(self):

        print('succeddful')
        test_true = self.X_test_all
        N, T, C = self.X_train_all.shape[0], self.X_train_all.shape[1], self.X_train_all.shape[2]
        _, T1, C1 = self.X_val_all.shape[0], self.X_val_all.shape[1], self.X_val_all.shape[2]
        _, T2, C2 = self.X_test_all.shape[0], self.X_test_all.shape[1], self.X_test_all.shape[2]
        print(self.X_train_all[0,0:2,:],self.X_val_all[0,0:3,:],self.X_test_all[0,0:4,:])
        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            self.scaler_mean=np.load(filename_mean)
            self.scaler_std=np.load(filename_std)
            X_train_all_norm = (self.X_train_all - self.scaler_mean) / self.scaler_std
            X_val_all_norm = (self.X_val_all - self.scaler_mean) / self.scaler_std
            X_test_all_norm = (self.X_test_all - self.scaler_mean) / self.scaler_std
            # 保存归一化副本，供窗口级插值使用
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
            '''X_train_all_fill=fill_4d_with_spline_vectorized(X_train_all_norm,train_mask)'''

            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_e_norm = (self.X_train_all_e - self_mean_e) / self_std_e
            X_val_all_e_norm = (self.X_val_all_e - self_mean_e) / self_std_e
            X_test_all_e_norm = (self.X_test_all_e - self_mean_e) / self_std_e

            #print(f'格点最终形状:{self_mean_e.shape},{self_std_e.shape},{X_train_all_e_norm.shape},{X_val_all_e_norm.shape},{X_test_all_e_norm.shape}')
        #print(f'数据集长度{T}{T1}{T2}')
        # 生成可重现的随机矩阵（使用现代化方法）
        #print('save success')
        #print("random_matrix hash:", get_array_hash(random_matrix_test))
        rng = np.random.default_rng(43)#42msl 43tmp 44u 45v
        #print(f'Random matrix shape: {random_matrix.shape}')
        #print(f'Random matrix unique values: {np.unique(random_matrix)}')
        if self.set_type == 0:
            filename=f'./mask_rate/{self.args.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                train_masks = np.load(filename)
            train_masks = np.array(train_masks)  # [10,N,T,1 or C]
        elif self.set_type == 1:
            filename=f'./mask_rate/{self.args.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                val_masks = np.load(filename)
            val_masks = np.array(val_masks)
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
            print(f"加载文件名{filename}")
            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                test_masks = np.load(filename)
                print( test_masks.shape)
            test_masks = np.array(test_masks)
        df_date = pd.read_csv('./data_provider/split_date.csv')
        df_stamp = df_date[['DATE']]
        df_stamp['DATE'] = pd.to_datetime(df_stamp['DATE'])  # 转换为datetime格式

        # 提取时间特征
        df_stamp['YEAR'] = df_stamp['DATE'].dt.year  # 年份（如 2023）
        df_stamp['MONTH'] = df_stamp['DATE'].dt.month  # 月份（1-12）
        df_stamp['WEEKDAY'] = df_stamp['DATE'].dt.weekday
        df_stamp['HOUR'] = df_stamp['DATE'].dt.hour
        date_co = df_stamp[['YEAR', 'MONTH', 'WEEKDAY', 'HOUR']].values
        print(date_co.shape)
        if self.set_type == 0:
            self.data_x_e = X_train_all_e_norm
            self.data_y = X_train_all_norm
            self.date_=date_co[0:12000,:]
            self.mask=train_masks
        elif self.set_type == 1:
            self.data_x_e = X_val_all_e_norm
            self.data_y = X_val_all_norm
            self.date_ = date_co[12000:13680, :]
            self.mask=val_masks
        else:
            self.data_x_e = X_test_all_e_norm
            self.data_y = test_true
            self.date_ = date_co[13680:, :]
            self.mask=test_masks
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y,
                                                                                  self.args)
            #print('ssdsdooedmoc,dlc,l')
        #print("数据增强 hash:", get_array_hash(self.data_x), get_array_hash(self.data_y))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # 窗口级 4D 稳健插值（mask 同 X 形状）
        o_mask = self.mask[:,:, s_begin:s_end, :]
        seq_x_e = self.data_x_e[:, s_begin:s_end, :]
        # 选择归一化基底
        if self.set_type == 0:
            base_norm = self.X_train_all_norm
        elif self.set_type == 1:
            base_norm = self.X_val_all_norm
        else:
            base_norm = self.X_test_all_norm
        window_norm = base_norm[:, s_begin:s_end, :]  # [N,24,C]
        filled_list = []
        min_points_cubic = getattr(self.args, 'min_points_cubic', 4)
        min_points_linear = getattr(self.args, 'min_points_linear', 2)
        for i in range(o_mask.shape[0]):
            filled = safe_fill_4d_with_fallback(
                window_norm, o_mask[i, :, :, :],
                clip_value=5.0,
                min_points_cubic=min_points_cubic,
                min_points_linear=min_points_linear
            )
            filled_list.append(filled)
        seq_x = np.stack(filled_list, axis=0)  # [10,N,24,C]
        seq_y = self.data_y[:, s_begin:s_end, :]
        s = self.locations
        e = self.locations_e
        datee=self.date_[s_begin:s_end, :]
        #print('日期形状')
        #print(datee.shape)
        #print(datee)
        # seq_x_mark = self.data_stamp[s_begin:s_end,:]
        # print(f'datemask的维度{seq_x_mark.shape}')

        return seq_x, seq_x_e, seq_y, s, e,datee,o_mask

    def __len__(self):
        return len(self.data_y[0]) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean
class Dataset_Custom_s0(Dataset):
    def __init__(self, args, root_path, data_path='ETTh1.csv',flag='train', set_size=None,
                 features='S', target='OT', timeenc=0, freq='h', seasonal_patterns=None,
                 X_train_all=None, X_val_all=None, X_test_all=None, locations=None,
                 X_train_all_e=None, X_val_all_e=None, X_test_all_e=None, locations_e=None,scale=True):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if set_size == None:
            self.seq_len = 24 * 1
            self.label_len = 0
            self.pred_len = 0
        else:
            self.seq_len = set_size[0]
            self.label_len = set_size[1]
            self.pred_len = set_size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.mask_rate = args.mask_rate
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.node_num=args.node
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_mean=None
        self.scaler_std=None
        self.__read_data__()

    def __read_data__(self):
        '''
        将maskN T C 每次获取1和 TC 参与以下运算
        '''
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        num_train, num_val, num_test = 12000, 1680, 3840
        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        X_train_all=df_data[border1s[0]:border2s[0]].values
        X_val_all = df_data[border1s[1]:border2s[1]].values
        X_test_all = df_data[border1s[2]:border2s[2]].values
        self.trues = df_data[border1s[2]:border2s[2]].values
        #print('训练验证测试')
        #print(X_train_all.shape,X_val_all.shape,X_test_all.shape)
        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            self.scaler_mean = np.load(filename_mean)
            self.scaler_std = np.load(filename_std)
            print(self.scaler_mean,self.scaler_std)
            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_norm=X_train_all.copy()
            X_val_all_norm=X_val_all.copy()
            X_test_all_norm=X_test_all.copy()
            X_train_all_norm[:,4]=(X_train_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_train_all_norm[:, 0:4] = (X_train_all_norm[:, 0:4] - self_mean_e) / self_std_e
            X_val_all_norm[:,4]=(X_val_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_val_all_norm[:, 0:4] = (X_val_all_norm[:, 0:4] - self_mean_e) / self_std_e
            X_test_all_norm[:,4]=(X_test_all_norm[:,4]-self.scaler_mean) / self.scaler_std
            X_test_all_norm[:, 0:4] = (X_test_all_norm[:, 0:4] - self_mean_e) / self_std_e
            # 持久化归一化序列，供 __getitem__ 窗口级插值使用
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
        features=1
        #print(f'Random matrix shape: {random_matrix.shape}')
        #print(f'Random matrix unique values: {np.unique(random_matrix)}')
        if self.set_type == 0:
            filename=f'./mask_rate/{self.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            train_masks = np.load(filename)
            print(f"加载文件名{filename}")
            print("random_matrix hash:", get_array_hash(train_masks[:,self.node_num,:,:]))
            train_masks=train_masks[:, self.node_num, :, :]
            train_masks = np.array(train_masks)  # shape: [10, T, C]
        elif self.set_type == 1:
            filename = f'./mask_rate/{self.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            val_masks = np.load(filename)
            print(f"加载文件名{filename}")
            print("random_matrix hash:", get_array_hash(val_masks[:, self.node_num, :, :]))
            val_masks = val_masks[:, self.node_num, :, :]
            val_masks = np.array(val_masks)  # shape: [10, T, C]
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}.npy'
            print(f"加载文件名{filename}")

            if os.path.exists(filename):
                print(f"Loading existing mask from {filename}")
                test_masks = np.load(filename)
                test_masks=test_masks[:, self.node_num, :, :]
                print( test_masks.shape)#10 tc
            test_masks = np.array(test_masks)

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['DATE']][border1:border2]

        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['DATE'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_y = X_train_all_norm
            self.mask=train_masks
        elif self.set_type == 1:
            self.data_y = X_val_all_norm
            self.mask=val_masks
        else:
            self.data_y = X_test_all
            self.mask=test_masks

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 窗口级插值：对 10 个掩码在 24 步窗口内动态填补
        o_mask = self.mask[:, s_begin:s_end, :] # 10 24 1
        # 选择归一化基底序列
        if self.set_type == 0:
            base_norm = self.X_train_all_norm
        elif self.set_type == 1:
            base_norm = self.X_val_all_norm
        else:
            base_norm = self.X_test_all_norm

        window_norm = base_norm[s_begin:s_end, :]  # [24, C] #y
        filled_list = []
        for i in range(o_mask.shape[0]):
            filled = window_norm * o_mask[i, :, :]

            filled_list.append(filled)

        seq_x = np.stack(filled_list, axis=0)  # [10, 24, C]
        seq_y = self.data_y[s_begin:s_end,:]
        seq_x_mark = self.data_stamp[s_begin:s_end,:]
        seq_y_mark = self.data_stamp[s_begin:s_end,:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,o_mask

    def __len__(self):
        return len(self.data_y) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean