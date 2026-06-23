import pandas as pd
from matplotlib import pyplot as plt
from pandas.tests.frame.test_validate import dataframe
from torch import Generator
from data_provider.data_factory_2 import data_provider
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
import seaborn as sns
import matplotlib.pyplot as plt

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
        wandb.watch(model, log="all")
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
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pos_s = pos_s.float().to(self.device)
                pos_e = pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                current_mask = mask.float().to(self.device)
                outputs,_,_,_,_= self.model(batch_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
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
                batch_y=batch_y.float().to(self.device)
                pos_s=pos_s.float().to(self.device)
                pos_e=pos_e.float().to(self.device)
                datee=datee.float().to(self.device)
                mask=mask.float().to(self.device)

                outputs,out,_,_,_= self.model(batch_x, batch_x_e, pos_s, pos_e, mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = mask[:, :, :, f_dim:]
                loss = criterion(outputs[current_mask == 0], batch_y_c[current_mask == 0])+criterion(out[current_mask == 0], batch_y_c[current_mask == 0])
                loss.backward(retain_graph=True)
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
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            mtime = os.path.getmtime(checkpoint_path)
            print(f"Checkpoint 最后修改时间: {time.ctime(mtime)}")  # 或使用 datetime 更规范
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        final_hash = hash_state_dict(self.model.state_dict(), device='cpu')
                # 获取 temporal 分支的 diag_lambda
        temporal_lambda = self.model.self_attention_layer[0]['temporal'].inner_attention.diag_lambda.item()

        # 获取 spatial 分支的 diag_lambda
        spatial_lambda = self.model.self_attention_layer[0]['spatial'].inner_attention.diag_lambda.item()

        # 记录到 wandb summary（推荐用于最终值）
        wandb.summary.update({
            "final_diag_lambda/temporal": temporal_lambda,
            "final_diag_lambda/spatial": spatial_lambda
        })
        print("Model Final Hash:", final_hash)
        preds = []
        trues = []
        masks = []
        attn_maps_T = []
        attn_maps_N = []
        attn_maps_F = []

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
                batch_x_e = batch_x_e.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pos_s = pos_s.float().to(self.device)
                pos_e = pos_e.float().to(self.device)
                datee = datee.float().to(self.device)
                current_mask=mask.float().to(self.device)
                # 模型前向传播
                outputs,_,atten_T,atten_N,atten_F= self.model(batch_x, batch_x_e, pos_s, pos_e, current_mask, datee)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                # add support for MS
                batch_y_c = batch_y[:, :, :, f_dim:]
                current_mask = current_mask[:, :, :, f_dim:]
                """if i==0:
                    attn_maps_T.append(atten_T.cpu())
                    attn_maps_N.append(atten_N.cpu())
                    attn_maps_F.append(atten_F.cpu())
                """
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true =batch_y_c.detach().cpu().numpy()
                current_mask=current_mask.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(current_mask)
                print(i)
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        B,N,T,C= trues.shape
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
        y1 = trues.reshape(-1, N, T * C)
        y2 = preds.reshape(-1, N, T * C)
        print(y1.shape,y2.shape)
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
            # ========================
            # 处理注意力图（仅 i==0）
            # ========================
        output_dir = 'attention_images'
        os.makedirs(output_dir, exist_ok=True)
        # 假设该代码段位于模型训练的验证/测试逻辑中
        if attn_maps_T:  # 确保有数据
            # 拼接数据
            attn_maps_T = torch.cat(attn_maps_T, dim=0)  # (B, H, L, S)
            attn_maps_N = torch.cat(attn_maps_N, dim=0)
            attn_maps_F = torch.cat(attn_maps_F, dim=0)
            B, H, N, _ = attn_maps_N.shape  # 假设是 (B, H, N, N)

            # 1. 初始化一个字典，用于收集本次所有要记录的数据
            current_epoch = getattr(self.args, 'current_epoch', 0)
            log_data = {"epoch": current_epoch}

            # 限制遍历的样本数，防止日志过大
            num_samples = min(4, B)

            for i in range(num_samples):
                attn_sample = attn_maps_N[i]  # (H, N, N)

                # ===== 1. 画平均图 (Combined Attention) =====
                attn_combined = attn_sample.mean(dim=0)  # (N, N)
                fig_comb, ax_comb = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    attn_combined.cpu().numpy(),
                    cmap="Reds",
                    square=True,
                    ax=ax_comb,
                    vmin=0,
                    vmax=1,
                    cbar=True,
                    cbar_kws={"shrink": 0.8}
                )
                ax_comb.set_title(f"Combined Spatial Attention (Sample {i}) - Epoch {current_epoch}")
                ax_comb.set_xlabel("Target Site (Key)")
                ax_comb.set_ylabel("Source Site (Query)")
                fig_comb.tight_layout()

                # 将图片保存到本地（可选）并放入 log 字典
                # image_path = os.path.join(output_dir, f"attention_combined_sample_{i}.png")
                # plt.savefig(image_path)
                log_data[f"attention_combined/sample_{i}"] = wandb.Image(fig_comb)
                plt.close(fig_comb)  # 及时关闭，释放内存

                # ===== 2. 画每个 head 的图 (Individual Heads) =====
                n_heads_to_plot = min(H, 8)  # 最多画 8 个 head
                n_cols = min(4, n_heads_to_plot)
                n_rows = (n_heads_to_plot + n_cols - 1) // n_cols

                fig_heads, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

                # 处理只有一个 head 的情况，确保 axes 是数组
                if n_heads_to_plot == 1:
                    axes_list = [axes]
                else:
                    axes_list = axes.flatten()

                for h in range(n_heads_to_plot):
                    ax = axes_list[h]
                    sns.heatmap(
                        attn_sample[h].cpu().numpy(),
                        cmap="Reds",
                        square=True,
                        ax=ax,
                        vmin=0,
                        vmax=1,
                        cbar=False
                    )
                    ax.set_title(f"Head {h}")
                    ax.set_xlabel("Target")
                    ax.set_ylabel("Source")

                # 隐藏多余的子图
                for h in range(n_heads_to_plot, len(axes_list)):
                    axes_list[h].axis('off')

                fig_heads.suptitle(f"All Heads Spatial Attention (Sample {i}) - Epoch {current_epoch}", fontsize=14)
                fig_heads.tight_layout(rect=[0, 0, 1, 0.96])

                # 将多头图放入 log 字典
                # image_path_heads = os.path.join(output_dir, f"attention_heads_sample_{i}.png")
                # plt.savefig(image_path_heads)
                log_data[f"attention_heads/sample_{i}"] = wandb.Image(fig_heads)
                plt.close(fig_heads)

            # 3. 核心改进：在循环外统一 log
            # 这确保了 sample_0 到 sample_9 的所有图片都在同一个 Step 下
            wandb.log(log_data)
            print(f"✅ Successfully logged attention maps for {num_samples} samples to W&B.")

        else:
            print("⚠️ No attention maps collected (i==0 not reached or model doesn't output them)")
        


        return mae, mse, rmse, smape,maape,r,r2


def plot_attention_maps(
        attn_weights,
        save_dir="./attention_figs",
        sample_idx=0,
        heads_to_plot=None,
        cmap="viridis",
        title_prefix="Attention",
        xlabel="Key Position",
        ylabel="Query Position",
        return_fig=False  # 新增：是否返回 figure 对象
):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import numpy as np

    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().numpy()

    B, H, L, S = attn_weights.shape
    assert sample_idx < B, f"sample_idx {sample_idx} >= batch size {B}"

    if heads_to_plot is None:
        heads_to_plot = list(range(H))
    else:
        heads_to_plot = [h for h in heads_to_plot if h < H]

    os.makedirs(save_dir, exist_ok=True)

    if len(heads_to_plot) == 1:
        h = heads_to_plot[0]
        attn_map = attn_weights[sample_idx, h]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(attn_map, cmap=cmap, square=True, ax=ax)
        ax.set_title(f"{title_prefix} (Sample {sample_idx}, Head {h})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        save_path = os.path.join(save_dir, f"attn_sample{sample_idx}_head{h}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None

    else:
        n_heads = len(heads_to_plot)
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_heads == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, h in enumerate(heads_to_plot):
            ax = axes[idx]
            attn_map = attn_weights[sample_idx, h]
            sns.heatmap(attn_map, ax=ax, cmap=cmap, square=True, cbar=True)
            ax.set_title(f"Head {h}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"{title_prefix} – Sample {sample_idx}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(save_dir, f"attn_sample{sample_idx}_heads{len(heads_to_plot)}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-head plot: {save_path}")

        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None

