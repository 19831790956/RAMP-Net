import numpy as np
# Author: Qidong Yang & Jonathan Giezendanner

import torch
from torch import nn as nn
from torch_geometric.data import Data
from utils.Activations import Tanh
from layers.GNN_Layer_External import GNN_Layer_External
from layers.GNN_Layer_Internal import GNN_Layer_Internal
from torch_geometric.nn import knn_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_passing = 4
        n_node_features_m = 24
        n_node_features_e = 24
        n_out_features = 1
        hidden_dim = 128
        self.feature = args.feature
        self.n_neighbors_m2m = 4
        self.n_node_features_m = n_node_features_m
        self.n_node_features_e = n_node_features_e
        self.n_passing = n_passing
        self.hidden_dim = hidden_dim
        self.n_out_features = n_out_features
        self.seq_len = args.seq_len
        self.out_len = args.seq_len

        self.gnn_ex_1 = GNN_Layer_External(in_dim=self.hidden_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                           ex_in_dim=self.n_node_features_e)
        self.gnn_ex_2 = GNN_Layer_External(in_dim=self.hidden_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
                                           ex_in_dim=self.n_node_features_e)

        self.gnn_layers = nn.ModuleList(modules=(
            GNN_Layer_Internal(
                in_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                org_in_dim=self.n_node_features_m)
            for _ in range(self.n_passing)))

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.n_node_features_m + 2, self.hidden_dim),
            Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Tanh())

        self.output_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        Tanh(),
                                        nn.Linear(self.hidden_dim, self.out_len * self.n_out_features))

    def build_graph_internal(self, x, madis_lon, madis_lat, edge_index):
        n_batch = x.size(0)

        n_stations = x.size(1)

        # (n_batch, n_stations, n_times * 3)#3是三个特征，风向风速温度
        x = x.reshape(n_batch * n_stations, -1)
        # (n_batch * n_stations, n_times * 3)

        pos = torch.cat((madis_lon, madis_lat), dim=2)

        # (n_batch, n_stations, 2)
        pos = pos.view(n_batch * n_stations, -1)
        # (n_batch * n_stations, 2)

        batch = torch.arange(n_batch).view(-1, 1) * torch.ones(1, n_stations)
        # (n_batch, n_stations)
        batch = batch.view(n_batch * n_stations, ).to(x.device)
        # (n_batch * n_stations, )

        index_shift = (torch.arange(n_batch) * n_stations).view(-1, 1, 1).to(x.device)
        edge_index = torch.cat(list(edge_index + index_shift), dim=1)
        # (2, n_batch * n_edges)

        graph = Data(x=x, pos=pos, batch=batch.long(), edge_index=edge_index.long())

        return graph

    def build_graph_external(self, madis_x, ex_x, ex_lon, ex_lat, edge_index):
        # madis_x: (n_batch, n_stations_m, n_features_m)
        # madis_lon: (n_batch, n_stations_m, 1)
        # madis_lat: (n_batch, n_stations_m, 1)
        # ex_x: (n_batch, n_stations_e, n_features_e)
        # ex_lon: (n_batch, n_stations_e, 1)
        # ex_lat: (n_batch, n_stations_e, 1)

        n_batch = madis_x.size(0)
        n_stations_m = madis_x.size(1)
        n_stations_e = ex_x.size(1)
        ex_x = ex_x.view(n_batch * n_stations_e, -1)
        # (n_batch * n_stations_e, n_features_e)
        # print('ssssssssssssssss')
        # print(n_batch, n_stations_m, n_stations_e)

        ex_pos = torch.cat((ex_lon.view(n_batch, n_stations_e, 1), ex_lat.view(n_batch, n_stations_e, 1)), dim=2)
        # (n_batch, n_stations_e, 2)
        ex_pos = ex_pos.view(n_batch * n_stations_e, -1)
        # (n_batch * n_stations_e, 2)
        '''每个批次偏移量'''
        madis_shift = (torch.arange(n_batch) * n_stations_m).view((n_batch, 1))
        ex_shift = (torch.arange(n_batch) * n_stations_e).view((n_batch, 1))

        shift = torch.cat((ex_shift, madis_shift), dim=1).unsqueeze(-1).to(madis_x.device)  # B 2 1
        edge_index = torch.cat(list(edge_index + shift), dim=1)
        # 第一行是madis 第二行是era
        # (2, n_batch * n_edges)

        graph = Data(x=ex_x, pos=ex_pos, edge_index=edge_index.long())

        return graph

    # 站点邻接矩阵 k=5
    # [madis j]
    # [madis i]
    def BuildMadisNetwork(self, lon, lat):  # N 1  N 1
        pos = torch.cat([lon, lat], dim=1)  # N 2
        k_edge_index = knn_graph(pos, k=self.n_neighbors_m2m, batch=torch.zeros((len(pos),)), loop=False)  # 所有站点属于一张图
        # k_edge_index = torch.tensor([[104], [105]]).to(self.device)
        return k_edge_index

    # [era]
    # [madis]
    def search_k_neighbors(self, base_points, cand_points, k):  # K=8  sta->era
        # base_points: (n_b, n_features)
        # cand_points: (n_c, n_features)

        # dis = torch.sum((base_points.unsqueeze(1) - cand_points.unsqueeze(0)) ** 2, dim=-1)# Nb Nc
        # 转弧度
        lat_b = torch.deg2rad(base_points[:, 0]).unsqueeze(1)  # (103, 1)
        lon_b = torch.deg2rad(base_points[:, 1]).unsqueeze(1)  # (103, 1)
        lat_c = torch.deg2rad(cand_points[:, 0]).unsqueeze(0)  # (1, 1682)
        lon_c = torch.deg2rad(cand_points[:, 1]).unsqueeze(0)  # (1, 1682)

        # Haversine 距离（单位：km，但只用于排序）
        dlat = lat_c - lat_b  # (103, 1682)
        dlon = lon_c - lon_b  # (103, 1682)
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_b) * torch.cos(lat_c) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        dis = 6371.0 * c  # (103, 1682)

        _, inds = torch.topk(dis, k, dim=1, largest=False)  # 每个站点对应的的网格点

        n_b = base_points.size(0)

        j_inds = inds.view((1, -1))
        i_inds = (torch.arange(n_b).view((-1, 1)) * torch.ones((n_b, k))).view((1, -1)).to(base_points.device)

        edge_index = torch.cat([j_inds, i_inds], dim=0)

        return edge_index


    def forward(self, obs_his, era_his, csta, cera, mask, datee):
        '''
                Input:obs_his:(B,Ns,L,C)
                      era_his:(B,Ne,L,C)
                      pan_fut:(B,C,lat,lon,L)
                      csta:(Ns,2)
                      cera:(Ne,2)
                '''
        # print('输入模型形状')
        # print(obs_his.size(), era_his.size(), csta.size(), cera.size())

        B, N, L, C = obs_his.shape
        csta = csta.cpu().numpy()
        cera = cera.cpu().numpy()
        obs_his_1 = obs_his
        obs_his_2 = obs_his
        filename_mean = f"./scaler/{self.feature}_mean.npy"
        filename_std = f"./scaler/{self.feature}_std.npy"
        scaler_mean = torch.from_numpy(np.load(filename_mean)).float().to(obs_his.device)  # [N]
        scaler_std = torch.from_numpy(np.load(filename_std)).float().to(obs_his.device)
        num_valid = torch.sum(mask == 1, dim=2)  # [B, N, C]
        eps = 1e-8
        num_valid = num_valid.clamp(min=eps)
        num_valid = num_valid.unsqueeze(2)  # BN1C
        # partly
        mean_enc_part = torch.sum(obs_his_2 * (mask == 1), dim=2).unsqueeze(2) / num_valid  # bnc
        mean_enc_part = mean_enc_part.detach()  # bn1c
        obs_his_2 = obs_his_2 - mean_enc_part
        std_enc_part = torch.sqrt(torch.sum((obs_his_2 * (mask == 1)) ** 2, dim=2).unsqueeze(2) / num_valid + 1e-5)
        std_enc_part = std_enc_part.detach()
        has_valid = (num_valid > eps)
        # 判断如果
        mean_enc_o = torch.where(
            has_valid,  # 条件：[B,1,n]
            mean_enc_part,  # 满足时：局部均值
            scaler_mean.expand_as(mean_enc_part)  # 不满足时：全局均值
        )  # bn1c
        std_enc_o = torch.where(
            has_valid,  # 条件：[B, N, 1, C]
            std_enc_part,  # 满足时：局部均值
            scaler_std.expand_as(std_enc_part)  # 不满足时：全局均值
        )  # bn1c
        obs_his = obs_his - mean_enc_o
        obs_his = obs_his.masked_fill(mask == 0, 0)
        obs_his = obs_his / std_enc_o

        mask_all_ones = torch.ones_like(era_his)
        mean_enc_e = torch.sum(era_his * (mask_all_ones == 1), dim=2) / torch.sum(mask_all_ones == 1, dim=2)
        mean_enc_e = mean_enc_e.unsqueeze(2).detach()
        era_his = era_his - mean_enc_e
        std_enc_e = torch.sqrt(
            torch.sum((era_his * (mask_all_ones == 1)) ** 2, dim=2) / torch.sum(mask_all_ones == 1, dim=2) + 1e-5)
        std_enc_e = std_enc_e.unsqueeze(2).detach()
        era_his /= std_enc_e
        csta = torch.tensor(csta).float().to(obs_his.device)
        cera = torch.tensor(cera).float().to(obs_his.device)

        B, NS, L, C = obs_his.shape
        B, NE, L, C = era_his.shape
        madis_x = obs_his  # B N L C
        # print(f'obs_his shape: {obs_his.shape},madis_x.shape: {madis_x.shape}')

        madis_point = csta.view(-1, 2)
        cand_point = cera.view(-1, 2)
        # (N,2

        madis_lon = csta.unsqueeze(0).repeat(B, 1, 1)[:, :, [1]]
        madis_lat = csta.unsqueeze(0).repeat(B, 1, 1)[:, :, [0]]

        # print(madis_lon.shape, madis_lat.shape)
        # B Ns 1

        ex_lon = cera[:, [1]].unsqueeze(0).repeat(B, 1, 1).view(B, -1)  # B N*1
        ex_lat = cera[:, [0]].unsqueeze(0).repeat(B, 1, 1).view(B, -1)

        # ex_x_era = era_his.permute(0, 2, 3, 4, 1).reshape(B, lat * lon, L, C)
        # B N L C

        ex_x = era_his.view(B, NE, -1)  # B N  L*C

        # lon_data:(N,1)
        lon_data = csta[:, [1]]
        lat_data = csta[:, [0]]
        # print('mmmm')
        # print(lon_data.size(), lat_data.size())

        edge_index = self.BuildMadisNetwork(lon_data, lat_data)  # station edge
        # 2 103*4
        # print('eeem')
        # print(madis_point.size(),cand_point.size())

        edge_index_e2m = self.search_k_neighbors(madis_point, cand_point, k=4)  # 站点经纬度，二维，era经纬度2维
        # 2  103*4

        n_batch, n_stations_m, n_hours_m, n_features_m = madis_x.shape  # BNLC

        madis_x = madis_x.reshape(n_batch, n_stations_m, -1)  # B N L*C 16 103 36

        in_graph = self.build_graph_internal(madis_x, madis_lon, madis_lat, edge_index)
        # 内网的batch*n作为batch x是数据 pos是这么多节点经纬度 edge是这么多节点边
        u = in_graph.x  # 16*103 36
        in_pos = in_graph.pos  # 16*103 2
        batch = in_graph.batch  # 16*103
        edge_index_m2m = in_graph.edge_index  # 2 16*103*5
        # 2, n_batch * n_stations * n_neighbours
        # 构造内部图之后，原本特性+pos生成新的特征
        # print(f'u和pos：{u.shape}——{in_pos.shape}')
        in_x = self.embedding_mlp(torch.cat((u, in_pos), dim=-1))

        if ex_x is not None:
            ex_graph = self.build_graph_external(madis_x, ex_x, ex_lon, ex_lat, edge_index_e2m)  # B N tc
            ex_x = ex_graph.x  # 16*1681 36
            ex_pos = ex_graph.pos  # 16*1681 2
            edge_index_e2m = ex_graph.edge_index  # 2 16*103*8
        if ex_x is not None:
            in_x = self.gnn_ex_1(in_x, ex_x, in_pos, ex_pos, edge_index_e2m, batch)  # 16*103 128
            # hidden_dim feature_e, 2, 2,NS*Kn
        # (n_batch * n_stations, hidden_dim)
        for i in range(self.n_passing):
            in_x = self.gnn_layers[i](in_x, u, in_pos, edge_index_m2m, batch)  # in_x是更新后的 u 是原本的
        # (n_batch * n_stations, hidden_dim)

        if ex_x is not None:
            in_x = self.gnn_ex_2(in_x, ex_x, in_pos, ex_pos, edge_index_e2m, batch)
        # (n_batch * n_stations, hidden_dim)
        out = self.output_mlp(in_x)  # 16*103 36
        # print(out.size())
        # (n_batch * n_stations, L*C)
        out = out.reshape(n_batch, n_stations_m, self.seq_len, self.n_out_features)  #
        out = out.mul(
            (std_enc_o[:, :, 0, :].unsqueeze(2).repeat(
                1, 1, self.out_len, 1)))
        out = out.add(
            (mean_enc_o[:, :, 0, :].unsqueeze(2).repeat(
                1, 1, self.out_len, 1)))
        # (B,N, L, C)
        return out,out