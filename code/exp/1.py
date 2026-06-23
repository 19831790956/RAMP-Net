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

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.loss=args.loss

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
        if self.loss == 'Huber':
            criterion = nn.SmoothL1Loss()
        elif self.loss == 'MSE':
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                '''mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                '''
                mask_y = torch.rand((B, T, 1)).to(self.device)
                mask_y[mask_y <= self.args.mask_rate] = 0  # masked
                mask_y[mask_y > self.args.mask_rate] = 1  # remained
                mask_else = torch.ones((B, T, N - 1)).to(self.device)
                mask = torch.cat((mask_else, mask_y), dim=-1)
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

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
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_losses = []
        vali_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                B, T, N = batch_x.shape
                # random mask
                '''
                mask = torch.rand((B, T, 1)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)
                '''
                mask_y = torch.rand((B, T, 1)).to(self.device)
                mask_y[mask_y <= self.args.mask_rate] = 0  # masked
                mask_y[mask_y > self.args.mask_rate] = 1  # remained
                mask_else = torch.ones((B, T, N - 1)).to(self.device)
                mask = torch.cat((mask_else,mask_y), dim=-1)
                inp = batch_x.masked_fill(mask == 0, 0)
                '''
                其他插补
                '''

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                '''if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                '''

                loss.backward()
                model_optim.step()

            '''print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))'''
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = f"{self.args.feature}_{self.args.loss}_{self.args.data_path}.png"  # 构建文件名
        picture_dir = "./picture"  # 图片保存的文件夹
        full_path = os.path.join(picture_dir, filename)
        plt.savefig(full_path)
        plt.show()

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
        data_stamps = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_x_o) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_o=batch_x_o.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask_y = torch.rand((B, T, 1)).to(self.device)
                mask_y[mask_y <= self.args.mask_rate] = 0  # masked
                mask_y[mask_y > self.args.mask_rate] = 1  # remained
                mask_else = torch.ones((B, T, N - 1)).to(self.device)
                mask = torch.cat((mask_else,mask_y), dim=-1)
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]
                batch_x_o = batch_x_o[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x_o.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + \
                             pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))


        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        shape = preds.shape
        print('test shape:', preds.shape, trues.shape, masks.shape)
        print(preds[0, 0:2, :], trues[0, 0:2, :])
        print(preds[32, 2:4, :], trues[32, 2:4, :])
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                print("!=")
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds = test_data.inverse_transform(preds.reshape(shape[0] * shape[1], -1)).reshape(shape)


        # result save
        folder_path = './results/' +self.args.feature+self.args.loss+ setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print('评估\n')
        print(preds[0, 0:2, :], trues[0, 0:2, :])
        print(preds[32, 2:4, :], trues[32, 2:4, :])
        mae, mse, rmse, smape,maape,r,r2 = metric(preds[masks == 0], trues[masks == 0])

        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        f = open("result_imputation.txt", 'a')
        f.write(self.args.feature+self.args.loss+setting + "  \n")
        f.write('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return mae, mse, rmse, smape,maape,r,r2

        '''
        def process_predictions(data_index,pred, true, pred_scale, true_scale,mask):
            # 将 pred 和 true 的最后一列添加到 DataFrame 中
            data_index['target'] = pred[:,-1]
            data_index['true'] = true[:,-1]
            data_index['pred_scale'] = pred_scale[:,-1]
            data_index['true_scale'] = true_scale[:,-1]
            data_index['mask'] = mask
            # 按 DATE 分组，并对 target 和 true 列求平均
            data_index['target'] = data_index.groupby('DATE')['target'].transform('mean')
            data_index['true'] = data_index.groupby('DATE')['true'].transform('mean')
            data_index['pred_scale'] = data_index.groupby('DATE')['pred_scale'].transform('mean')
            data_index['true_scale'] = data_index.groupby('DATE')['true_scale'].transform('mean')
            data_index['mask'] = data_index.groupby('DATE')['mask'].transform('mean')
            # 去除重复的行
            fdata_index =  data_index.drop_duplicates(subset=['DATE'])

            # 提取处理后的值
            preds_y = fdata_index['target'].values
            trues_y = fdata_index['true'].values
            pred_scale_y = fdata_index['pred_scale'].values
            true_scale_y = fdata_index['true_scale'].values
            masks_y = fdata_index['mask'].values

            # 返回处理后的值和缩放因子
            return preds_y, trues_y, pred_scale_y, true_scale_y,masks_y
        print('只选取y')

        print(f'预测值：{preds.shape},真实值true1:{trues.shape},掩码:{mask_y.shape}')
        print(data_stamps[0:100])
        preds_y, trues_y, pred_scale_y, true_scale_y,mask_y=process_predictions(data_index, preds, trues, pred_scale, true_scale,mask_y)
        print('最终17520')
        print(f'预测值：{preds.shape},真实值true1:{trues.shape},掩码:{mask_y.shape}')
        df = pd.DataFrame({'Predictions': preds_y, 'True Values': trues_y})
        df.to_csv('./predictions_vs_true_values.csv', index=False)
        print('以及写入文件')
        mae, mse, rmse, smape,maape,r,r2 = metric(preds_y[mask_y== 0], trues_y[mask_y == 0])
        mae_s, mse_s, rmse_s, smape_s, maape_s, r_s, r2_s = metric(pred_scale_y[mask_y == 0], true_scale_y[mask_y== 0])
        print('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        print('mae_s:{},mse_s:{},rmse_s:{},smape_s:{},maape_s:{},r_s:{},r2_s:{}'.format(mae_s, mse_s, rmse_s, smape_s, maape_s, r_s, r2_s))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{},rmse:{},smape:{},maape:{},r:{},r2:{}'.format(mae, mse, rmse, smape,maape,r,r2))
        f.write('\n')
        f.write('mae_s:{},mse_s:{},rmse_s:{},smape_s:{},maape_s:{},r_s:{},r2_s:{}'.format(mae_s, mse_s, rmse_s, smape_s, maape_s, r_s, r2_s))
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return mae, mse, rmse, smape,maape,r,r2
        '''
