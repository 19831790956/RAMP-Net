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
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(vali_loader):
                # random mask
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape = batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                #print(f'batch_x{batch_x.shape}{mask.shape}')
                current_x = batch_x
                current_mask = mask
                # 模型前向传播
                outputs,out = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                out = out[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        vali_steps = len(vali_loader)
        print(f'训练的大小{vali_steps}')
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
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(
                flag='train',
                train_generator=train_generator,
                base_seed=base_seed  # ← 关键：复用 generator
            )
            train_steps = len(train_loader)
            print(f'训练的大小{train_steps}')
            iter_count = 0
            train_loss = []
            #mask_idx = epoch % 10  # 每10个epoch循环使用mask
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(train_loader):
                if i %100== 0:
                    print(i)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                batch_y=batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee=datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape=batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                current_x = batch_x
                current_mask = mask


                outputs, out = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                out = out[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                loss.backward(retain_graph=True)
                model_optim.step()
                train_loss.append(loss.item())
                """if i%200==0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/epoch_step": epoch + i / len(train_loader),
                        "epoch": epoch + i / len(train_loader)
                    })"""
            train_loss = np.average(train_loss)
            print('训练成功')
            print('ssssssssssss')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({
                "all_train/loss_epoch_avg": train_loss,
                "all_val/loss_epoch_avg": vali_loss,
                "epoch": epoch  # 整数 epoch
            })  # 用 epoch 当横轴更清晰
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch , train_steps, train_loss, vali_loss))
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)
            # 在每个 epoch 训练结束后添加：
            print(f"Epoch {epoch} - Model state hash: {hash_state_dict(self.model.state_dict())}")

        print(len(train_losses),len(vali_losses))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = f"{self.modelname}_{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}_{self.args.mask_rate}.png"  # 构建文件名
        picture_dir = "./picture"  # 图片保存的文件夹
        full_path = os.path.join(picture_dir, filename)
        plt.savefig(full_path)
        plt.show()
        self.vail_loss=np.mean(vali_losses)

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
        #with open('./all_weights.txt', 'w') as f:
            #f.write('')
        self.model.eval()
        test_log={}
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape = batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                #print(f'batch_x{batch_x.shape}{mask.shape}')
                B,N,T,C=batch_x.shape
                current_x = batch_x
                current_mask = mask
                # 模型前向传播
                outputs, _ = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true =batch_y_c.detach().cpu().numpy()
                current_mask=current_mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(current_mask)
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        shape = trues.shape
        print('test shape:', preds.shape, trues.shape,masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds=preds.transpose(1, 0, 2, 3)
            preds=preds.reshape(shape[1],-1,shape[-1])
            print('guiyihua ')
            print(preds.shape)#N BT C
            preds = test_data.inverse_transform(preds)
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
        f.write(
            'model:{},mask:{},vail_loss:{},mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, maape:{:.4f}, r:{:.4f}, r2:{:.4f}\n'.format(
                self.modelname, self.args.mask_rate, self.vail_loss, mae, mse, rmse, smape, maape, r, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds[masks == 0])
        np.save(folder_path + 'true.npy', trues[masks == 0])
        zmask = masks.reshape(-1, N, T * C)
        wandb_images = []
        # 创建保存可视化结果的文件夹
        real_pred_folder = './real_pred'+self.modelname
        if not os.path.exists(real_pred_folder):
            os.makedirs(real_pred_folder)
        for node in [0, 10]:
            for loc in range(0,1200,120):
                print(loc)
                plt.figure(figsize=(10, 5))
                plt.plot(y1[loc, node, :], label='True')
                plt.plot(y2[loc, node, :], label='Pred')
                plt.xlabel('Time')
                plt.ylabel('value')
                plt.title('True and Pred ')
                plt.legend()
                # 标记 mask == 0 的点
                for i in np.where(zmask[loc, node, :] == 0)[0]:
                    plt.scatter(i, y1[loc, node, :][i], c='red', marker='o', edgecolor='black', zorder=5)
                    plt.scatter(i, y2[loc, node, :][i], c='blue', marker='o', edgecolor='black', zorder=5)
                    plt.text(i, y1[loc, node, :][i], f'{y1[loc, node, :][i]:.2f}', fontsize=8, ha='right', va='bottom')
                    plt.text(i, y2[loc, node, :][i], f'{y2[loc, node, :][i]:.2f}', fontsize=8, ha='left', va='top')
                filename = f"{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}_{self.args.mask_rate}_{node}_{loc}.png"  # 构建文件名
                full_path = os.path.join(real_pred_folder, filename)
                plt.savefig(full_path, dpi=150, bbox_inches='tight')
                #wandb_img = wandb.Image(plt.gcf(), caption=f"Node {node}, Sample {loc}")
                #wandb_images.append(wandb_img)
                plt.close()  # 避免内存泄漏
        #wandb.log({"test_predictions": wandb_images})

        return mae, mse, rmse, smape,maape,r,r2
"""

返回一个变量的
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
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(vali_loader):
                # random mask
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape = batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                #print(f'batch_x{batch_x.shape}{mask.shape}')
                current_x = batch_x
                current_mask = mask
                # 模型前向传播
                outputs, _ = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                #loss = criterion(pred[mask == 0], true[mask == 0])
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        vali_steps = len(vali_loader)
        print(f'训练的大小{vali_steps}')
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
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(
                flag='train',
                train_generator=train_generator,
                base_seed=base_seed  # ← 关键：复用 generator
            )
            train_steps = len(train_loader)
            print(f'训练的大小{train_steps}')
            iter_count = 0
            train_loss = []
            #mask_idx = epoch % 10  # 每10个epoch循环使用mask
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(train_loader):
                if i%100 == 0:
                    print(i)
                    final_hash = hash_state_dict(self.model.state_dict(), device='cpu')
                    print("Model Final Hash:", final_hash)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                batch_y=batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee=datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape=batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                current_x = batch_x
                current_mask = mask


                outputs, _ = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())
                if i%200==0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/epoch_step": epoch + i / len(train_loader),
                        "epoch": epoch + i / len(train_loader)
                    })
            train_loss = np.average(train_loss)
            print('训练成功')
            print('ssssssssssss')
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({
                "all_train/loss_epoch_avg": train_loss,
                "all_val/loss_epoch_avg": vali_loss,
                "epoch": epoch  # 整数 epoch
            })  # 用 epoch 当横轴更清晰
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch , train_steps, train_loss, vali_loss))
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)
            # 在每个 epoch 训练结束后添加：
            print(f"Epoch {epoch} - Model state hash: {hash_state_dict(self.model.state_dict())}")

        print(len(train_losses),len(vali_losses))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(vali_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        filename = f"{self.modelname}_{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}_{self.args.mask_rate}.png"  # 构建文件名
        picture_dir = "./picture"  # 图片保存的文件夹
        full_path = os.path.join(picture_dir, filename)
        plt.savefig(full_path)
        plt.show()
        self.vail_loss=np.mean(vali_losses)

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
        #with open('./all_weights.txt', 'w') as f:
            #f.write('')
        self.model.eval()
        test_log={}
        with torch.no_grad():
            for i, (batch_x, batch_x_e, batch_y, pos_s,pos_e,datee,mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y=batch_y.unsqueeze(1)
                batch_y=batch_y.repeat(1,10,1,1,1)
                batch_x_e = batch_x_e.float().to(self.device)
                batch_x_e=batch_x_e.unsqueeze(1)
                batch_x_e=batch_x_e.repeat(1,10,1,1,1)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                datee=datee.unsqueeze(1)
                datee=datee.repeat(1,10,1,1)
                mask=mask.float().to(self.device)
                pos_s=pos_s[0]
                pos_e=pos_e[0]
                shapee = batch_x.shape
                batch_x=batch_x.reshape(-1,shapee[2],shapee[3],shapee[4])
                shape = batch_x_e.shape
                batch_x_e= batch_x_e.reshape(-1,shape[2],shape[3],shape[4])
                batch_y=batch_y.reshape(-1,shapee[2],shapee[3],shapee[4])
                mask=mask.reshape(-1,shapee[2],shapee[3],shapee[4])
                shapedate=datee.shape
                datee=datee.reshape(-1,shapedate[2],shapedate[3])
                #print(f'batch_x{batch_x.shape}{mask.shape}')
                B,N,T,C=batch_x.shape
                current_x = batch_x
                current_mask = mask
                # 模型前向传播
                outputs, _ = self.model(current_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true =batch_y_c.detach().cpu().numpy()
                current_mask=current_mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(current_mask)
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        shape = trues.shape
        print('test shape:', preds.shape, trues.shape,masks.shape)
        if test_data.scale and self.args.inverse:
            shape = trues.shape
            if preds.shape[-1] != trues.shape[-1]:
                preds = np.tile(preds, [1, 1, int(trues.shape[-1] / preds.shape[-1])])
            preds=preds.transpose(1, 0, 2, 3)
            preds=preds.reshape(shape[1],-1,shape[-1])
            print('guiyihua ')
            print(preds.shape)#N BT C
            preds = test_data.inverse_transform(preds)
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
        f.write(
            'model:{},mask:{},vail_loss:{},mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, maape:{:.4f}, r:{:.4f}, r2:{:.4f}\n'.format(
                self.modelname, self.args.mask_rate, self.vail_loss, mae, mse, rmse, smape, maape, r, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, smape,maape,r,r2]))
        np.save(folder_path + 'pred.npy', preds[masks == 0])
        np.save(folder_path + 'true.npy', trues[masks == 0])
        zmask = masks.reshape(-1, N, T * C)
        wandb_images = []
        # 创建保存可视化结果的文件夹
        real_pred_folder = './real_pred'+self.modelname
        if not os.path.exists(real_pred_folder):
            os.makedirs(real_pred_folder)
        for node in [0, 10]:
            for loc in range(0,1200,120):
                print(loc)
                plt.figure(figsize=(10, 5))
                plt.plot(y1[loc, node, :], label='True')
                plt.plot(y2[loc, node, :], label='Pred')
                plt.xlabel('Time')
                plt.ylabel('value')
                plt.title('True and Pred ')
                plt.legend()
                # 标记 mask == 0 的点
                for i in np.where(zmask[loc, node, :] == 0)[0]:
                    plt.scatter(i, y1[loc, node, :][i], c='red', marker='o', edgecolor='black', zorder=5)
                    plt.scatter(i, y2[loc, node, :][i], c='blue', marker='o', edgecolor='black', zorder=5)
                    plt.text(i, y1[loc, node, :][i], f'{y1[loc, node, :][i]:.2f}', fontsize=8, ha='right', va='bottom')
                    plt.text(i, y2[loc, node, :][i], f'{y2[loc, node, :][i]:.2f}', fontsize=8, ha='left', va='top')
                filename = f"{self.args.feature}_{self.args.loss}_{self.args.learning_rate}_{self.args.d_model}_{self.args.batch_size}_{self.args.mask_rate}_{node}_{loc}.png"  # 构建文件名
                full_path = os.path.join(real_pred_folder, filename)
                plt.savefig(full_path, dpi=150, bbox_inches='tight')
                #wandb_img = wandb.Image(plt.gcf(), caption=f"Node {node}, Sample {loc}")
                #wandb_images.append(wandb_img)
                plt.close()  # 避免内存泄漏
        #wandb.log({"test_predictions": wandb_images})

        return mae, mse, rmse, smape,maape,r,r2


"""

