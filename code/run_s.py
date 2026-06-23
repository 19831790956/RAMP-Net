import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_DETERMINISTIC'] = '1'
print("✅ 正在运行 run.py 文件：", __file__)

# ----------- 必须最顶部设定全局seed：-------------
import argparse
import pandas as pd
import json
import wandb
parser = argparse.ArgumentParser(description='MPNN')
# ... 这里保留你的各种 parser.add_argument ...（如原代码）
parser.add_argument('--seed', type=int, default=2024, help='random seed for full reproducibility')
parser.add_argument('--node', type=int, default=0, help='node ')

parser.add_argument('--task_name', type=str, required=False, default='imputation',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='weather_mask_0.25', help='model id')
parser.add_argument('--model', type=str, required=False, default='TSATT2',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

parser.add_argument('--feature', type=str, default='TMP', help='File name for the output CSV file')
parser.add_argument('--loss', type=str, default='Huber', help='loss')
parser.add_argument('--way', type=str, default='everynode', help='标准化方式')
# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')

parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--iitr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# GPU
parser.add_argument('--use_gpu', action='store_true', help='use GPU', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seEEed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")


# TimeXer
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
# 动态参数
parser.add_argument('--root_path', type=str, default='./all_data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='72595524259.csv', help='data file')
parser.add_argument('--start_itr', type=int, default=0, help='从第几个实验开始')
parser.add_argument('--resume', action='store_true', help='从断点恢复训练')
parser.add_argument('--resume_epoch', type=int, default=0, help='从指定epoch恢复（配合--resume使用）')
args = parser.parse_args()

BASE_SEED = args.seed
os.environ['PYTHONHASHSEED'] = str(BASE_SEED)

import random
import numpy as np
import torch
torch.set_num_threads(1)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(BASE_SEED)
    torch.cuda.manual_seed_all(BASE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_anomaly_enabled(True)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(BASE_SEED)

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation_l import Exp_Imputation_l
from exp.exp_imputation_s import Exp_Imputation_s
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
if __name__ == '__main__':

    with open('../metadata_5K.json', 'r') as f:
        meta_data = json.load(f)

    data_csv_dir = '../all_data_' + args.feature  # B 的 CSV 文件存储目录

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        if args.model=='LR':
        	Exp = Exp_Imputation_l
        else:
        	Exp = Exp_Imputation_s
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast
    project_name=f'{args.model}_{args.feature}{args.mask_rate}_{args.seed}'
    wandb.init(project='52_Guangdong_Cali', name=project_name, mode='offline')
    print(args.mask_rate)
    os.makedirs(f'./results_show/{args.model}{args.feature}', exist_ok=True)



    # 仍然保留 results（不破坏你原有逻辑）
    results = []
    if args.is_training:
        # 定义结果文件
        result_file = f"./results_show/{args.model}{args.feature}/{args.model}_{args.feature}_{args.loss}_{args.batch_size}_{args.learning_rate}_{args.d_model}_{args.mask_rate}_{args.iitr}.csv"
        print(f'读取结果文件{result_file}')
        # >>> 第一步：读取已处理的站点（实现断点续跑）<<<
        done_stations = set()
        if os.path.exists(result_file):
            try:
                df_done = pd.read_csv(result_file)
                print('try')
                df_done = df_done.dropna()
                done_stations = set(df_done['station'].astype(str))
                print(f"✅ 已加载 {len(done_stations)} 个已完成的站点，将跳过它们。")
                print(f"已完成站点示例: {list(done_stations)[:5]}")
            except Exception as e:
                print(f"⚠️  读取历史结果文件失败: {e}，将重新开始。")
                done_stations = set()
        else:
            print(f"📁 未发现结果文件 {result_file}，将从头开始处理所有站点。")

            # 如果是训练模式且 itr > 1，我们只记录最终测试结果，所以可以安全跳过
            # 如果是测试模式，也一样处理

            # 用于追加写入结果
        file_mode = 'a' if os.path.exists(result_file) else 'w'
        # 确保表头存在
        if file_mode == 'w':
            with open(result_file, 'w') as f:
                f.write("station,idx,mae,mse,rmse,smape,maape,r,r2,val_loss\n")
        # >>> 遍历所有站点，跳过已完成的 <<<
        sorted_items = sorted(meta_data.items())  # 按 station 名排序
        for idx, (station_file, station_info) in enumerate(sorted_items):
                # 提取 station 名（去掉路径或后缀，确保匹配）
                station_name = station_file  # 假设 station_file 是唯一标识，如 'A1.csv'
                station_id = station_file.replace('.csv', '')
                # >>> 关键：跳过已完成的站点 <<<
                
                if station_name in done_stations:
                    print(f"⏭️  跳过已处理站点: {station_name}")
                    continue  # 跳过，不处理
                
                print(f"🔄 开始处理新站点: {station_name,idx}")

                args.root_path = data_csv_dir
                args.data_path = station_file
                args.node =idx  # count 仅用于 node 编号，不影响逻辑
                exp = Exp(args)
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_ls{}_ft{}_dt{}_lr{}_batch{}_{}_{}_st{}_mk{}'.format(
                    args.task_name, args.model_id, args.model, args.data,
                    args.features, args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers,
                    args.d_ff, args.expand, args.d_conv, args.factor,
                    args.embed, args.loss, args.feature, args.distil,
                    args.learning_rate, args.batch_size, args.des, args.iitr,  # ✅ 使用 ii 表示第几次实验
                    station_id, args.mask_rate  # ✅ 加在最后，作为唯一标识
                )

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                _, train_loss, val_loss = exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mae, mse, rmse, smape, maape, r, r2 = exp.test(setting)

                # >>> 写入文件（追加）<<<
                with open(result_file, 'a') as f:
                    f.write(f"{station_name},{args.node},{mae},{mse},{rmse},{smape},{maape},{r},{r2},{val_loss}\n")

                results.append({
                    'station': station_name,
                    'idx': args.node,
                    'val_loss': val_loss,
                    'test_loss_mae': mae,
                    'test_loss_mse': mse,
                    'test_loss_rmse': rmse,
                    'test_loss_smape': smape,
                    'test_loss_maape': maape,
                    'test_loss_r': r,
                    'test_loss_r2': r2
                })

                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()

    else:
            # 定义结果文件
            os.makedirs('./test_1', exist_ok=True)
            result_file = f"./test_1/{args.model}_{args.feature}_{args.loss}_{args.batch_size}_{args.learning_rate}_{args.d_model}_{args.mask_rate}_{args.iitr}.csv"
            done_stations = set()
            if os.path.exists(result_file):
                try:
                    df_done = pd.read_csv(result_file)
                    print('try')
                    df_done = df_done.dropna()
                    done_stations = set(df_done['station'].astype(str))
                    print(f"✅ 已加载 {len(done_stations)} 个已完成的站点，将跳过它们。")
                    print(f"已完成站点示例: {list(done_stations)[:5]}")
                except Exception as e:
                    print(f"⚠️  读取历史结果文件失败: {e}，将重新开始。")
                    done_stations = set()
            else:
                print(f"📁 未发现结果文件 {result_file}，将从头开始处理所有站点。")

                # 如果是训练模式且 itr > 1，我们只记录最终测试结果，所以可以安全跳过
                # 如果是测试模式，也一样处理

                # 用于追加写入结果
            file_mode = 'a' if os.path.exists(result_file) else 'w'
            # 确保表头存在
            if file_mode == 'w':
                with open(result_file, 'w') as f:
                    f.write("station,idx,mae,mse,rmse,smape,maape,r,r2,val_loss\n")
            # >>> 遍历所有站点，跳过已完成的 <<<
            # 改成排序遍历
            sorted_items = sorted(meta_data.items())  # 按 station 名排序
            for idx, (station_file, station_info) in enumerate(sorted_items):

                # 提取 station 名（去掉路径或后缀，确保匹配）
                station_name = station_file  # 假设 station_file 是唯一标识，如 'A1.csv'
                station_id = station_file.replace('.csv', '')
                # >>> 关键：跳过已完成的站点 <<<
                if station_name in done_stations:
                    print(f"⏭️  跳过已处理站点: {station_name}")
                    continue  # 跳过，不处理

                print(f"🔄 开始处理新站点: {station_name, idx}")

                args.root_path = data_csv_dir
                args.data_path = station_file
                args.node = idx  # count 仅用于 node 编号，不影响逻辑
                exp = Exp(args)


                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_ls{}_ft{}_dt{}_lr{}_batch{}_{}_{}_st{}_mk{}'.format(
                    args.task_name, args.model_id, args.model, args.data,
                    args.features, args.seq_len, args.label_len, args.pred_len,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers,
                    args.d_ff, args.expand, args.d_conv, args.factor,
                    args.embed, args.loss, args.feature, args.distil,
                    args.learning_rate, args.batch_size, args.des, args.iitr,
                    station_id, args.mask_rate  # ✅ 加在最后，作为唯一标识
                )

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mae, mse, rmse, smape, maape, r, r2 = exp.test(setting, test=1)

                # ✅ val_loss 在测试时未定义，应设为 'nan' 或从模型获取（若无，则用 nan）
                val_loss = 'nan'  # 因为没有训练，val_loss 不存在

                with open(result_file, 'a') as f:
                    f.write(f"{station_name},{args.node},{mae},{mse},{rmse},{smape},{maape},{r},{r2},{val_loss}\n")

                results.append({
                    'station': station_name,
                    'idx': args.node,
                    'val_loss': val_loss,
                    'test_loss_mae': mae,
                    'test_loss_mse': mse,
                    'test_loss_rmse': rmse,
                    'test_loss_smape': smape,
                    'test_loss_maape': maape,
                    'test_loss_r': r,
                    'test_loss_r2': r2
                })

                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()


    # =============================================
    # ✅ 最终统计：从文件读取所有结果（包括之前和本次新增的）
    # =============================================

    result_file = f"./results_show/{args.model}{args.feature}/{args.model}_{args.feature}_{args.loss}_{args.batch_size}_{args.learning_rate}_{args.d_model}_{args.mask_rate}_{args.iitr}.csv"
    if os.path.exists(result_file):
            results_df_from_file = pd.read_csv(result_file)
            print(f'\n📊 从文件 {result_file} 加载最终结果，共 {len(results_df_from_file)} 个站点')
            avg_val_loss = results_df_from_file['val_loss'].mean()
            avg_test_mae = results_df_from_file['mae'].mean()
            avg_test_mse = results_df_from_file['mse'].mean()
            avg_test_rmse = results_df_from_file['rmse'].mean()
            avg_test_smape = results_df_from_file['smape'].mean()
            avg_test_maape = results_df_from_file['maape'].mean()
            avg_test_r = results_df_from_file['r'].mean()
            avg_test_r2 = results_df_from_file['r2'].mean()
            print(f"✅ Average Val LOSS: {avg_val_loss:.4f}")
            print(f"✅ Average Test MAE: {avg_test_mae:.4f}")
            print(f"✅ Average Test MSE: {avg_test_mse:.4f}")
            print(f"✅ Average Test RMSE: {avg_test_rmse:.4f}")
            print(f"✅ Average Test SMAPE: {avg_test_smape:.4f}%")
            print(f"✅ Average Test MAAPE: {avg_test_maape:.4f}%")
            print(f"✅ Average Test R: {avg_test_r:.4f}")
            print(f"✅ Average Test R2: {avg_test_r2:.4f}")

            # 保存平均结果
            summary_file = f'average_losses_{args.feature}.txt'
            with open(summary_file, 'a') as f:
                f.write(f"{args.model}\n")
                f.write(f"{args.loss}\n")
                f.write(f"{args.batch_size}_{args.learning_rate}_{args.mask_rate}\n")
                f.write(f"Average Val LOSS: {avg_val_loss:.4f}\n")
                f.write(f"Average Test MAE: {avg_test_mae:.4f}\n")
                f.write(f"Average Test MSE: {avg_test_mse:.4f}\n")
                f.write(f"Average Test RMSE: {avg_test_rmse:.4f}\n")
                f.write(f"Average Test SMAPE: {avg_test_smape:.4f}%\n")
                f.write(f"Average Test MAAPE: {avg_test_maape:.4f}%\n")
                f.write(f"Average Test R: {avg_test_r:.4f}\n")
                f.write(f"Average Test R2: {avg_test_r2:.4f}\n")