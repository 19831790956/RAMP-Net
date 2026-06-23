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
class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.loss=args.loss
        self.modelname=args.model
        self.log_dir = './parameter/'
        self.vail_loss=0

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
        return 0

    '''def train(self, setting):

        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

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
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(flag='train', current_epoch=epoch)
            train_steps = len(train_loader)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_e, batch_y, pos_s, pos_e, datee, mask) in enumerate(train_loader):
                if i == 0:
                    batch_hash = get_batch_hash(batch_x)
                    hashes.append(batch_hash)
                    print(f"Epoch {epoch} first batch hash: {batch_hash}")
        return train_loss'''

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
        #test_data, test_loader = self._get_data(flag='test')

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

        return self.model,0,0


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
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
        #with open('./all_weights.txt', 'w') as f:
            #f.write('')
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y=batch_y.float().to(self.device)
                mask=mask.float().to(self.device)
                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_x = batch_x[:, :,:, f_dim:]

                # add support for MS
                mask = mask[:,:, :, f_dim:]
                batch_y=batch_y[:, :,:, f_dim:]
                #batch_x_o = batch_x_o[:, :, f_dim:]

                batch_x = batch_x.detach().cpu().numpy()
                pred = batch_x
                true = batch_y.detach().cpu().numpy()
                mask=mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask)

                '''if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))'''



        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        shape = trues.shape
        print('test shape:', preds.shape, trues.shape,masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds = test_data.inverse_transform(preds)
            print('guiyihua最终 ')
            print(preds.shape)
        # result save
        folder_path = './results/' +self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, smape,maape,r,r2 = metric(preds[masks == 0], trues[masks == 0])
        print(preds[0,0,:,:], trues[0,0,:,:])

        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        f = open("result_imputation.txt", 'a')
        f.write(self.args.feature+self.args.loss+setting + "  \n")
        f.write('3次样条插值——mask:{},vail_loss:{},mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(self.args.mask_rate,self.vail_loss,mae, mse, rmse, smape,maape,r,r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds[masks == 0])
        np.save(folder_path + 'true.npy', trues[masks == 0])


        return mae, mse, rmse, smape,maape,r,r2

