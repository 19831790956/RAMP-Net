import pandas as pd
from matplotlib import pyplot as plt
from pandas.tests.frame.test_validate import dataframe
from torch import Generator
from data_provider.data_factory import data_provider
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
class Exp_Imputation_s(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation_s, self).__init__(args)
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,mask) in enumerate(vali_loader):
                if i == 0:
                    batch_hash = get_batch_hash(batch_x)
                    print(f" first batch hash: {batch_hash}")
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.unsqueeze(1)
                batch_y = batch_y.repeat(1, 10, 1, 1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark = batch_x_mark.unsqueeze(1)
                batch_x_mark = batch_x_mark.repeat(1, 10, 1, 1)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.unsqueeze(1)
                batch_y_mark = batch_y_mark.repeat(1, 10, 1, 1)
                mask = mask.float().to(self.device)
                shapee = batch_x.shape
                batch_x = batch_x.reshape(-1, shapee[2], shapee[3])
                shapeey = batch_y.shape
                batch_y = batch_y.reshape(-1, shapeey[2], shapeey[3])
                shapexm = batch_x_mark.shape
                batch_x_mark = batch_x_mark.reshape(-1, shapexm[2], shapexm[3])
                shapeym = batch_y_mark.shape
                batch_y_mark = batch_y_mark.reshape(-1, shapeym[2], shapeym[3])
                mask = mask.reshape(-1, shapee[2], 1)

                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(
                    outputs[:, :, [4]][mask == 0],  # 仍在GPU上
                    batch_y[:, :, [4]][mask == 0]  # 仍在GPU上
                )
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        print("🔍 Debug Info:")
        print("args.seed =", self.args.seed, type(self.args.seed))
        print("num_workers =", self.args.num_workers)
        print("use_gpu =", getattr(self.args, 'use_gpu', False))
        if hasattr(self, 'device'):
            print("device =", self.device)

        # 检查 generator
        train_generator = Generator()
        train_generator.manual_seed(self.args.seed)
        print("Generator initial_seed =", train_generator.initial_seed())

        # 检查数据
        feature_dir = f"../{self.args.feature}"
        X_train_all = np.load(os.path.join(feature_dir, 'X_train_all.npy'))
        print("Data shape =", X_train_all.shape)
        print("Data mean =", X_train_all.mean())
        vali_data, vali_loader = self._get_data(flag='val')
        vail_steps = len(vali_loader)
        print(f'验证的大小{vail_steps }')
        path = os.path.join(self.args.checkpoints, setting)
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
        print("Train seed:", self.args.seed)
        print("Generator initial seed:", base_seed)  # 看看是不是你设的值
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(
                flag='train',
                train_generator=train_generator,
                base_seed=base_seed# ← 关键：复用 generator
            )
            train_steps = len(train_loader)
            print(f'训练的大小{train_steps}')
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,mask) in enumerate(train_loader):

                #print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape, mask.shape)
                if i == 0:
                    batch_hash = get_batch_hash(batch_x)
                    hashes.append(batch_hash)
                    print(f"Epoch {epoch} first batch hash: {batch_hash}")
                if i % 200 == 0:
                    print(i)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark=batch_x_mark.unsqueeze(1)
                batch_x_mark=batch_x_mark.repeat(1,10,1,1)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark=batch_y_mark.unsqueeze(1)
                batch_y_mark=batch_y_mark.repeat(1,10,1,1)
                mask = mask.float().to(self.device)
                shapee = batch_x.shape
                batch_x = batch_x.reshape(-1, shapee[2], shapee[3])
                shapeey = batch_y.shape
                batch_y = batch_y.reshape(-1, shapeey[2], shapeey[3])
                shapexm = batch_x_mark.shape
                batch_x_mark = batch_x_mark.reshape(-1, shapexm[2], shapexm[3])
                shapeym = batch_y_mark.shape
                batch_y_mark = batch_y_mark.reshape(-1, shapeym[2], shapeym[3])
                mask = mask.reshape(-1, shapee[2], 1)

                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[:,:,[4]][mask == 0], batch_y[:,:,[4]][mask == 0])

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


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model,np.mean(train_losses),np.mean(vali_losses)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_steps = len(test_loader)
        print(f'测试的大小{test_steps }')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        final_hash = hash_state_dict(self.model.state_dict(), device='cpu')
        print("Model Final Hash:", final_hash)
        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_stamps = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,mask) in enumerate(test_loader):
                if i == 0:
                    batch_hash = get_batch_hash(batch_x)
                    print(f"first batch hash: {batch_hash}")
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.unsqueeze(1)
                batch_y = batch_y.repeat(1, 10, 1, 1)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark = batch_x_mark.unsqueeze(1)
                batch_x_mark = batch_x_mark.repeat(1, 10, 1, 1)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.unsqueeze(1)
                batch_y_mark = batch_y_mark.repeat(1, 10, 1, 1)
                mask = mask.float().to(self.device)
                shapee = batch_x.shape

                batch_x = batch_x.reshape(-1, shapee[2], shapee[3])
                B, T, C = batch_x.shape
                shapeey = batch_y.shape
                batch_y = batch_y.reshape(-1, shapeey[2], shapeey[3])
                shapexm = batch_x_mark.shape
                batch_x_mark = batch_x_mark.reshape(-1, shapexm[2], shapexm[3])
                shapeym = batch_y_mark.shape
                batch_y_mark = batch_y_mark.reshape(-1, shapeym[2], shapeym[3])
                mask = mask.reshape(-1, shapee[2], 1)
                outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mask)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        shape = preds.shape
        print('test shape:', preds.shape, trues.shape, masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                print("!=")
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds = test_data.inverse_transform(preds.reshape(shape[0] * shape[1], -1)).reshape(shape)
        if self.node in [0, 10, 50, 100]:
            # 在这里的代码
            trues_1=trues[:,:,[4]]
            preds_1=preds[:,:,[4]]
            masks_1=masks[:,:,[0]]
            y1 = trues_1.reshape(-1, T * 1)
            y2 = preds_1.reshape(-1,T * 1)
            print(y1.shape,y2.shape)
            zmask = masks_1.reshape(-1,T * 1)
            # 创建保存可视化结果的文件夹
            real_pred_folder = './real_pred'+self.modelname
            if not os.path.exists(real_pred_folder):
                os.makedirs(real_pred_folder)
            for loc in range(0,1200,240):
                print(loc)
                plt.figure(figsize=(10, 5))
                plt.plot(y1[loc,:], label='True')
                plt.plot(y2[loc,:], label='Pred')
                plt.xlabel('Time')
                plt.ylabel('value')
                plt.title('True and Pred ')
                plt.legend()
                print(y1[loc,  :],y2[loc, :])
                # 标记 mask == 0 的点
                for i in np.where(zmask[loc,:] == 0)[0]:
                    plt.scatter(i, y1[loc,  :][i], c='red', marker='o', edgecolor='black', zorder=5)
                    plt.scatter(i, y2[loc, :][i], c='blue', marker='o', edgecolor='black', zorder=5)
                    plt.text(i, y1[loc,  :][i], f'{y1[loc,  :][i]:.2f}', fontsize=8, ha='right', va='bottom')
                    plt.text(i, y2[loc, :][i], f'{y2[loc,  :][i]:.2f}', fontsize=8, ha='left', va='top')
                filename = f"{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.mask_rate}_{self.args.batch_size}_{self.node}_{loc}.png"  # 构建文件名
                full_path = os.path.join(real_pred_folder, filename)
                plt.savefig(full_path)
                plt.show()
        # result save
        folder_path = './results/' +self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print('评估\n')

        mae, mse, rmse, smape,maape,r,r2 = metric(preds[:,:,[4]][masks == 0], trues[:,:,[4]][masks == 0])

        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))


        return mae, mse, rmse, smape,maape,r,r2

