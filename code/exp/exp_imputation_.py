import pandas as pd
from matplotlib import pyplot as plt
from pandas.tests.frame.test_validate import dataframe

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.loss=args.loss
        self.modelname=args.model
        self.log_dir = './parameter/'

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.SmoothL1Loss()
        return criterion



    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee) in enumerate(vali_loader):

                # random mask
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                B, N, T, C = batch_x.shape

                mask = torch.rand((B, N, T,C)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs,_= self.model(inp, batch_x_e, pos_s,pos_e,mask,datee)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :,:, f_dim:]
                batch_x = batch_x[:,:, :, f_dim:]
                mask = mask[:, :,:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_losses = []
        vali_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee=datee.float().to(self.device)
                B, N, T,C = batch_x.shape

                mask = torch.rand((B, N, T,C)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1

                inp = batch_x.masked_fill(mask == 0, 0)

                pos_s=pos_s[0]
                pos_e=pos_e[0]

                outputs,_ = self.model(inp, batch_x_e, pos_s,pos_e,mask,datee)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:,:, :, f_dim:]

                batch_x = batch_x[:,:, :, f_dim:]
                mask = mask[:,:, :, f_dim:]
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                self.print_parameters_and_gradients(epoch, i)

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model,np.average(train_losses),np.average(vali_losses)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y=batch_y.float().to(self.device)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                B, N, T,C = batch_x.shape
                mask = torch.rand((B, N, T,C)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1

                inp = batch_x.masked_fill(mask == 0, 0)
                outputs,weight_a = self.model(inp, batch_x_e, pos_s,pos_e,mask,datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :,:, f_dim:]
                mask = mask[:,:, :, f_dim:]
                batch_y=batch_y[:, :,:, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_y.detach().cpu().numpy()
                mask=mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask)

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape,masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds=preds.transpose(1, 0, 2, 3)
            preds=preds.reshape(shape[1],-1)
            print('guiyihua ')
            print(preds.shape)
            preds=preds.transpose()
            preds = test_data.inverse_transform(preds)
            preds=preds.transpose()
            preds=preds.reshape(shape[1],shape[0],shape[2],shape[3])
            preds=preds.transpose(1, 0, 2, 3)
            print('guiyihua最终 ')
            print(preds.shape)
        y1 = trues.reshape(-1, N, T * C)
        y2 = preds.reshape(-1, N, T * C)
        print(y1.shape,y2.shape)

        # result save
        folder_path = './results/' +self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print('评估\n')
        print(preds[0,0,:],trues[0,0,:])
        mae, mse, rmse, smape,maape,r,r2 = metric(preds[masks == 0], trues[masks == 0])

        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        f = open("result_imputation.txt", 'a')
        f.write(self.args.feature+self.args.loss+setting + "  \n")
        f.write('model:{},mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(self.modelname,mae, mse, rmse, smape,maape,r,r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds[masks == 0])
        np.save(folder_path + 'true.npy', tarues[masks == 0])


        return mae, mse, rmse, smape,maape,r,r2

