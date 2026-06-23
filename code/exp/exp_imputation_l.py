import pandas as pd
from matplotlib import pyplot as plt
from pandas.tests.frame.test_validate import dataframe
from torch import Generator
from data_provider.data_factory_1 import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import hashlib
import warnings
import numpy as np
import wandb
def get_batch_hash(batch_tensor):
    # 将 tensor 转为 numpy 并计算 hash
    batch_np = batch_tensor.detach().cpu().numpy()
    return hashlib.md5(batch_np.tobytes()).hexdigest()
def hash_state_dict(state_dict, device='cpu'):
        m = hashlib.sha256()

        # 按 key 排序确保一致性
        for key in sorted(state_dict.keys()):
            # 移动到指定设备再转为 numpy 数组
            tensor = state_dict[key].detach().to('cpu').numpy()
            # 转为 bytes
            t_bytes = tensor.tobytes()
            m.update(t_bytes)

        return m.hexdigest()
class Exp_Imputation_l(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_l, self).__init__(args)
        self.loss=args.loss
        self.modelname=args.model
        self.log_dir = './parameter/'
        self.node=args.node


    def _build_model(self):
        print('Building model...')
        model = self.model_dict[self.args.model].Model(self.args).float()
        initial_hash = hash_state_dict(model.state_dict(), device='cpu')
        print("Model Initial Hash:", initial_hash)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _get_data(self, flag,train_generator=None,base_seed=None):
        data_set, data_loader = data_provider(args=self.args, flag=flag,train_generator=train_generator,base_seed=base_seed)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.loss == 'Huber':
            criterion = nn.SmoothL1Loss()
        elif self.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        print('vail_begin')
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                mask = mask.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs,batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):

        # 检查 generator
        train_generator = Generator()
        train_generator.manual_seed(self.args.seed)

        # 检查数据
        feature_dir = f"../{self.args.feature}"
        X_train_all = np.load(os.path.join(feature_dir, 'X_train_all.npy'))
        print("Data mean =", X_train_all.mean())
        vali_data, vali_loader = self._get_data(flag='val')
        vail_steps = len(vali_loader)
        path = os.path.join(self.args.checkpoints, self.args.model + self.args.feature,setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_losses = []
        vali_losses = []
        hashes = []
        train_generator = Generator()
        train_generator.manual_seed(self.args.seed)  # 只设一次！
        base_seed = train_generator.initial_seed()
        for epoch in range(self.args.train_epochs):
            start_time = time.time()
            train_data, train_loader = self._get_data(
                flag='train',
                train_generator=train_generator,
                base_seed=base_seed# ← 关键：复用 generator
            )
            train_steps = len(train_loader)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, mask) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                mask = mask.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                # 反向传播
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)
            end_time = time.time()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = f"{self.args.feature}_{self.args.loss}_{self.modelname}_{self.args.node}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}.png"  # 构建文件名
        picture_dir = f"./picture/{self.modelname}"  # 图片保存的文件夹
        if not os.path.exists(picture_dir):
            os.makedirs(picture_dir)
        full_path = os.path.join(picture_dir, filename)

        plt.savefig(full_path)
        plt.show()
        return self.model,np.mean(train_losses),np.mean(vali_losses)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_steps = len(test_loader)
        print(f'测试的大小{test_steps }')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/'+self.args.model+self.args.feature+'/' + setting, 'checkpoint.pth')))
        final_hash = hash_state_dict(self.model.state_dict(), device='cpu')
        print("Model Final Hash:", final_hash)
        preds = []
        trues = []
        masks = []
        data_stamps = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                mask = mask.float().to(self.device)

                outputs = self.model(batch_x)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                masks.append(mask.detach().cpu())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks, axis=0)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            preds = test_data.inverse_transform(preds)
            print('test shape:', preds.shape, trues.shape, masks.shape)# 6 10  1 1

        # result save
        folder_path = './results_show/' +self.args.model+self.args.feature+'/'+self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, smape,maape,r,r2 = metric(preds[masks == 0], trues[masks == 0])

        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        preds[masks ==1]=trues[masks == 1]

        np.save(folder_path + 'mask.npy', masks)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return mae, mse, rmse, smape,maape,r,r2



