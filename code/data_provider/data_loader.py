import os
import numpy as np
import pandas as pd
from torch.nn.attention.flex_attention import or_masks

from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features

import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

import hashlib
def get_array_hash(array):
    return hashlib.md5(array.tobytes()).hexdigest()

class Dataset_Custom_l(Dataset):
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
        self.flag = flag
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


        if self.scale:
            filename_mean = f"./scaler/{self.args.feature}_mean.npy"
            filename_std = f"./scaler/{self.args.feature}_std.npy"
            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            if all(os.path.exists(f) for f in [filename_mean, filename_std, filename_mean_e, filename_std_e]):
                self.scaler_mean = np.load(filename_mean)
                self.scaler_std = np.load(filename_std)
                scaler_e=np.load(filename_mean_e)
                std_e=np.load(filename_std_e)
                print(self.scaler_mean,self.scaler_std,scaler_e,std_e)
                X_train_i = (X_train_i - scaler_e) / std_e
                X_train_o=(X_train_o - self.scaler_mean) / self.scaler_std
                X_val_i = (X_val_i - scaler_e) / std_e
                X_val_o = (X_val_o - self.scaler_mean) / self.scaler_std
                X_test_i = (X_test_i - scaler_e) / std_e
                X_test_o = X_test_o


            else:
                print("no exist")
        df_date = pd.read_csv('./data_provider/split_date.csv')
        all_dates = pd.to_datetime(df_date['DATE'])
        X_train_i = np.tile(X_train_i[np.newaxis, :, :], (10, 1, 1))
        X_train_o = np.tile(X_train_o[np.newaxis, :, :], (10, 1, 1))
        X_val_i = np.tile(X_val_i[np.newaxis, :, :], (10, 1, 1))
        X_val_o = np.tile(X_val_o[np.newaxis, :, :], (10, 1, 1))
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}_{self.args.iitr}.npy'
        test_masks = np.load(filename)
        test_mask=test_masks[:,self.node_num,:,:]#10 T 1
        X_test_i = np.tile(X_test_i[np.newaxis, :, :], (10, 1, 1))#10 T 4
        X_test_o= np.tile(X_test_o[np.newaxis, :, :], (10, 1, 1))  # 10 T 1
        if self.set_type == 0:
            self.data_x = X_train_i
            self.data_y = X_train_o
            self.mask = np.ones_like(X_train_o)
            self.raw_dates = all_dates.iloc[0:12000].reset_index(drop=True)
            self.valid_indices = list(range(len(self.raw_dates) - self.seq_len - self.pred_len + 1))
        elif self.set_type == 1:
            self.data_x = X_val_i
            self.data_y =  X_val_o
            self.mask = np.ones_like(X_val_o)
            self.raw_dates = all_dates.iloc[12000:13680].reset_index(drop=True)
            self.valid_indices = list(range(len(self.raw_dates) - self.seq_len - self.pred_len + 1))
        else:
            self.data_x = X_test_i
            self.data_y = X_test_o
            self.mask=test_mask
            self.raw_dates = all_dates.iloc[13680:].reset_index(drop=True)
            self.build_valid_indices()

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def build_valid_indices(self):

        self.valid_indices = []

        T = len(self.raw_dates)

        for s_begin in range(T - self.seq_len - self.pred_len + 1):

            s_end = s_begin + self.seq_len - 1

            start_month = self.raw_dates.iloc[s_begin].to_period('M')
            end_month   = self.raw_dates.iloc[s_end].to_period('M')

            if start_month == end_month:
                self.valid_indices.append(s_begin)

        self.export_valid_indices_csv(f"{self.args.model}{self.flag}_nocross")
    def export_valid_indices_csv(self, save_path):

        rows = []

        for sample_id, start_idx in enumerate(self.valid_indices):

            end_idx = start_idx + self.seq_len - 1

            start_date = self.raw_dates.iloc[start_idx]
            end_date   = self.raw_dates.iloc[end_idx]

            rows.append({
                'sample_id': sample_id,
                'global_idx': start_idx,
                'start_date': start_date,
                'end_date': end_date,
                'month': start_date.strftime('%Y-%m'),
                'cross_month': start_date.month != end_date.month
            })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

        print(f'Saved: {save_path}')
    def __getitem__(self, index):
        s_begin = self.valid_indices[index]
        s_end = s_begin + self.seq_len

        seq_x = self.data_x[:,s_begin:s_end, :]
        seq_y = self.data_y[:,s_begin:s_end,:]
        mask=self.mask[:,s_begin:s_end,:]
        return seq_x,seq_y,mask



    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean
"""标准化"""
class Dataset_Custom_0(Dataset):
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
        self.flag = flag
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
            print(filename_mean,filename_std)
            self.scaler_mean=np.load(filename_mean)
            self.scaler_std=np.load(filename_std)
            print(self.scaler_mean, self.scaler_std)
            X_train_all_norm = (self.X_train_all - self.scaler_mean) / self.scaler_std
            X_val_all_norm = (self.X_val_all - self.scaler_mean) / self.scaler_std
            X_test_all_norm = (self.X_test_all - self.scaler_mean) / self.scaler_std
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm

            filename_mean_e = f"./scaler/{self.args.feature}_mean_e.npy"
            filename_std_e = f"./scaler/{self.args.feature}_std_e.npy"
            self_mean_e=np.load(filename_mean_e)
            self_std_e=np.load(filename_std_e)
            X_train_all_e_norm = (self.X_train_all_e - self_mean_e) / self_std_e
            X_val_all_e_norm = (self.X_val_all_e - self_mean_e) / self_std_e
            X_test_all_e_norm = (self.X_test_all_e - self_mean_e) / self_std_e
        else:
            self.X_train_all_norm = self.X_train_all
            self.X_val_all_norm = self.X_val_all
            self.X_test_all_norm = self.X_test_all
            X_train_all_e_norm=self.X_train_all_e
            X_val_all_e_norm=self.X_val_all_e
            X_test_all_e_norm=self.X_test_all_e
        if self.set_type == 0:
            filename=f'./mask_rate/{self.args.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            if os.path.exists(filename):
                train_masks = np.load(filename)
            train_masks = np.array(train_masks)  # [10,N,T,1 or C]
        elif self.set_type == 1:
            filename=f'./mask_rate/{self.args.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            if os.path.exists(filename):
                val_masks = np.load(filename)
            val_masks = np.array(val_masks)
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}_{self.args.iitr}.npy'
            if os.path.exists(filename):
                test_masks = np.load(filename)
                print( test_masks.shape)
            test_masks = np.array(test_masks)
        df_date = pd.read_csv('./data_provider/split_date.csv')
        all_dates = pd.to_datetime(df_date['DATE'])
        df_stamp = df_date[['DATE']]
        data_stamp = time_features(pd.to_datetime(df_stamp['DATE'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        date_co = data_stamp[:,[0,3]]
        if self.set_type == 0:
            Z_temp = np.expand_dims(self.X_train_all_norm, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)
            self.data_y = Z_repeated.copy()
            Z_repeated[:, :, :,-1] = Z_repeated[:, :,:, -1] * train_masks.squeeze(-1)
            ##
            self.data_x=Z_repeated
            E_temp = np.expand_dims(X_train_all_e_norm, axis=0)
            E_repeated = np.repeat(E_temp, 10, axis=0)
            self.data_x_e = E_repeated
            date_train=date_co[0:12000,:]
            T_temp = np.expand_dims(date_train, axis=0)
            T_repeated = np.repeat(T_temp, 10, axis=0)
            self.date_=T_repeated
            self.raw_dates = all_dates.iloc[0:12000].reset_index(drop=True)
            self.mask=train_masks
            self.build_valid_indices()
        elif self.set_type == 1:
            Z_temp = np.expand_dims(self.X_val_all_norm, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)
            self.data_y = Z_repeated.copy()
            Z_repeated[:, :, :,-1] = Z_repeated[:, :,:, -1] * val_masks.squeeze(-1)
            ##
            self.data_x=Z_repeated
            E_temp = np.expand_dims(X_val_all_e_norm, axis=0)
            E_repeated = np.repeat(E_temp, 10, axis=0)
            self.data_x_e = E_repeated
            date_val=date_co[12000:13680, :]
            T_temp = np.expand_dims(date_val, axis=0)
            T_repeated = np.repeat(T_temp, 10, axis=0)
            self.date_=T_repeated
            self.raw_dates = all_dates.iloc[12000:13680].reset_index(drop=True)
            self.mask=val_masks
            self.build_valid_indices()
        else:
            Z_temp = np.expand_dims(self.X_test_all_norm, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)
            Z_repeated[:, :, :,-1] = Z_repeated[:, :,:, -1] * test_masks.squeeze(-1)
            ##
            self.data_x=Z_repeated
            Y_temp = np.expand_dims(test_true, axis=0)
            Y_repeated = np.repeat(Y_temp, 10, axis=0)
            self.data_y=Y_repeated
            E_temp = np.expand_dims(X_test_all_e_norm, axis=0)
            E_repeated = np.repeat(E_temp, 10, axis=0)
            self.data_x_e = E_repeated
            date_test=date_co[13680:, :]
            T_temp = np.expand_dims(date_test, axis=0)
            T_repeated = np.repeat(T_temp, 10, axis=0)
            self.date_=T_repeated
            self.raw_dates = all_dates.iloc[13680:].reset_index(drop=True)
            self.mask=test_masks
            self.build_valid_indices()
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y,
                                                                                  self.args)
    def build_valid_indices(self):

        self.valid_indices = []

        T = len(self.raw_dates)

        for s_begin in range(T - self.seq_len - self.pred_len + 1):

            s_end = s_begin + self.seq_len - 1

            start_month = self.raw_dates.iloc[s_begin].to_period('M')
            end_month   = self.raw_dates.iloc[s_end].to_period('M')

            if start_month == end_month:
                self.valid_indices.append(s_begin)

        print(f' {len(self.valid_indices)}')
        self.export_valid_indices_csv(f"{self.args.model}{self.flag}_nocross")

    def export_valid_indices_csv(self, save_path):

        rows = []

        for sample_id, start_idx in enumerate(self.valid_indices):

            end_idx = start_idx + self.seq_len - 1

            start_date = self.raw_dates.iloc[start_idx]
            end_date   = self.raw_dates.iloc[end_idx]

            rows.append({
                'sample_id': sample_id,
                'global_idx': start_idx,
                'start_date': start_date,
                'end_date': end_date,
                'month': start_date.strftime('%Y-%m'),
                'cross_month': start_date.month != end_date.month
            })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

        print(f'Saved: {save_path}')


    def __getitem__(self, index):
        s_begin = self.valid_indices[index]
        s_end = s_begin + self.seq_len
        seq_x=self.data_x[:,:,s_begin:s_end,:]
        seq_x_e=self.data_x_e[:,:,s_begin:s_end,:]
        seq_y=self.data_y[:,:,s_begin:s_end,:]
        s=self.locations
        e=self.locations_e
        datee=self.date_[:,s_begin:s_end,:]
        o_mask=self.mask[:,:,s_begin:s_end,:]
        return seq_x, seq_x_e, seq_y, s, e,datee,o_mask

    def __len__(self):
        return len(self.valid_indices)

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
        self.flag = flag
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

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        print(self.root_path,self.data_path)
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
            self.X_train_all_norm = X_train_all_norm
            self.X_val_all_norm = X_val_all_norm
            self.X_test_all_norm = X_test_all_norm
        else:
            self.X_train_all_norm = X_train_all
            self.X_val_all_norm = X_val_all
            self.X_test_all_norm = X_test_all

        if self.set_type == 0:

            filename=f'./mask_rate/{self.mask_rate}/train_mask_{self.args.feature}_{self.args.iitr}.npy'
            train_masks = np.load(filename)
            train_masks=train_masks[:, self.node_num, :, :]
            train_masks = np.array(train_masks)
        elif self.set_type == 1:
            filename = f'./mask_rate/{self.mask_rate}/val_mask_{self.args.feature}_{self.args.iitr}.npy'
            val_masks = np.load(filename)
            val_masks = val_masks[:, self.node_num, :, :]
            val_masks = np.array(val_masks)
        else:
            filename = f'./mask_rate/{self.args.mask_rate}/test_mask_{self.args.feature}_{self.args.iitr}.npy'

            if os.path.exists(filename):
                test_masks = np.load(filename)
                test_masks=test_masks[:, self.node_num, :, :]
            test_masks = np.array(test_masks)

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['DATE']][border1:border2]

        df_stamp['DATE'] = pd.to_datetime(df_stamp.DATE)
        df_date = pd.read_csv('./data_provider/split_date.csv')
        all_dates = pd.to_datetime(df_date['DATE'])
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
            ##
            Z_temp = np.expand_dims(self.X_train_all_norm, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)  #
            self.data_y = Z_repeated.copy()
            Z_repeated[:, :, -1] = Z_repeated[:, :, -1] * train_masks.squeeze(-1)
            ##
            self.data_x=Z_repeated
            full_mask = np.ones_like(self.data_x, dtype=train_masks.dtype)
            full_mask[:, :, [4]] = train_masks
            self.mask=full_mask
            self.raw_dates = all_dates.iloc[0:12000].reset_index(drop=True)
            self.valid_indices = list(range(len(self.raw_dates) - self.seq_len - self.pred_len + 1))
        elif self.set_type == 1:
            ##
            Z_temp = np.expand_dims(self.X_val_all_norm, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)
            self.data_y=Z_repeated.copy()
            Z_repeated[:, :, -1] = Z_repeated[:, :, -1] * val_masks.squeeze(-1)

            ##
            self.data_x = Z_repeated
            full_mask = np.ones_like(self.data_x, dtype=val_masks.dtype)
            full_mask[:, :, [4]] = val_masks
            self.mask = full_mask
            self.raw_dates = all_dates.iloc[12000:13680].reset_index(drop=True)
            self.valid_indices = list(range(len(self.raw_dates) - self.seq_len - self.pred_len + 1))
        else:
            ##
            Z_temp = np.expand_dims(X_test_all, axis=0)
            Z_repeated = np.repeat(Z_temp, 10, axis=0)
            self.data_y = Z_repeated.copy()
            X_temp = np.expand_dims(self.X_test_all_norm, axis=0)
            X_repeated = np.repeat(X_temp, 10, axis=0)
            X_repeated[:, :, -1] = X_repeated[:, :, -1] * test_masks.squeeze(-1)
            self.data_x = X_repeated
            full_mask = np.ones_like(self.data_x, dtype=test_masks.dtype)
            full_mask[:, :, [4]] = test_masks
            self.mask = full_mask
            self.raw_dates = all_dates.iloc[13680:].reset_index(drop=True)
            self.build_valid_indices()

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        date_= np.expand_dims(data_stamp, axis=0)
        self.data_stamp=np.repeat(date_, 10, axis=0)
    def build_valid_indices(self):

        self.valid_indices = []

        T = len(self.raw_dates)

        for s_begin in range(T - self.seq_len - self.pred_len + 1):

            s_end = s_begin + self.seq_len - 1

            start_month = self.raw_dates.iloc[s_begin].to_period('M')
            end_month   = self.raw_dates.iloc[s_end].to_period('M')
            if start_month == end_month:
                self.valid_indices.append(s_begin)

        print(f'{len(self.valid_indices)}')
        self.export_valid_indices_csv(f"{self.args.model}{self.flag}_nocross")

    def export_valid_indices_csv(self, save_path):

        rows = []

        for sample_id, start_idx in enumerate(self.valid_indices):

            end_idx = start_idx + self.seq_len - 1

            start_date = self.raw_dates.iloc[start_idx]
            end_date   = self.raw_dates.iloc[end_idx]

            rows.append({
                'sample_id': sample_id,
                'global_idx': start_idx,
                'start_date': start_date,
                'end_date': end_date,
                'month': start_date.strftime('%Y-%m'),
                'cross_month': start_date.month != end_date.month
            })

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')

        print(f'Saved: {save_path}')
    def __getitem__(self, index):
        s_begin = self.valid_indices[index]
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[:, s_begin:s_end,:]
        seq_y = self.data_y[:, s_begin:s_end,:]
        o_mask = self.mask[:, s_begin:s_end, :]
        seq_x_mark = self.data_stamp[:, s_begin:s_end,:]
        seq_y_mark = self.data_stamp[:, s_begin:s_end,:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,o_mask

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return data*self.scaler_std+self.scaler_mean
