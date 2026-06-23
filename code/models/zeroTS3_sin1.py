import os
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from networkx.classes import neighbors
from scipy.interpolate import interp1d
from torch.nn.functional import layer_norm
from sklearn.neighbors import NearestNeighbors
from utils.masking import TriangularCausalMask
from math import sqrt, sin


#实现在N上参数共享，在C不参数共享，在L上卷积的TCN
class TCN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,num_layers=3,dilation=1):
        super(TCN, self).__init__()
        layers=[]
        self.num_layers = num_layers
        self.start_conv=nn.Conv2d(in_channels,out_channels,kernel_size=(1,kernel_size),dilation=(1,dilation),
                          padding=(0,(kernel_size - 1) * dilation // 2))
        for i in range(num_layers):
            layers.append(nn.Conv2d(out_channels,out_channels,kernel_size=(1,kernel_size),dilation=(1,dilation),
                          padding=(0,(kernel_size - 1) * dilation // 2)))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            dilation*=2#膨胀因子每次加倍

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        #input:B,C,N,L
        #output:B,D,N,L
        x=self.start_conv(x)
        residual = x
        h=[]
        for layer in self.tcn:
            x=layer(x)
            h.append(x)
        x = x + residual
        return x

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class AttentionLayer_T(nn.Module):#8*12
    def __init__(self, attention, d_model, n_heads, q_dmodel,k_dmodel,v_dmodel,d_keys=None,
                 d_values=None):
        super(AttentionLayer_T, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(q_dmodel, d_keys * n_heads)#64 8 8
        self.key_projection = nn.Linear(k_dmodel, d_keys * n_heads)
        self.value_projection = nn.Linear(v_dmodel, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads,v_dmodel)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        Bq, Dq, Nq, Lq= queries.shape
        Bk, Dk, Nk, Lk= keys.shape
        Bv, Dv, Nv, Lv = values.shape
        H = self.n_heads
        queries=queries.permute(0,2,3,1)# B N L D
        keys=keys.permute(0,2,3,1)
        values=values.permute(0,2,3,1)
        queries=queries.reshape(Bq*Nq,Lq,Dq)
        keys=keys.reshape(Bk*Nk,Lk,Dk)
        values=values.reshape(Bv*Nv,Lv,Dv)
        queries = self.query_projection(queries).reshape(Bq*Nq,Lq,H,-1)# BNLD
        keys = self.key_projection(keys).reshape(Bk*Nk,Lk,H,-1)#
        values = self.value_projection(values).reshape(Bv*Nv,Lv,H,-1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        shapeo=out.shape
        out=out.reshape(shapeo[0],shapeo[1],-1)
        out=out.reshape(Bq,Nq,Lq,-1)
        return self.out_projection(out), attn
class AttentionLayer_N(nn.Module):
    def __init__(self, attention, d_model, n_heads, q_dmodel,k_dmodel,v_dmodel,d_keys=None,
                 d_values=None):
        super(AttentionLayer_N, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(q_dmodel, d_keys * n_heads)#64 8 8
        self.key_projection = nn.Linear(k_dmodel, d_keys * n_heads)
        self.value_projection = nn.Linear(v_dmodel, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads,v_dmodel)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        Bq, Dq, Nq, Lq= queries.shape
        Bk, Dk, Nk, Lk= keys.shape
        Bv, Dv, Nv, Lv = values.shape
        H = self.n_heads
        queries=queries.permute(0,3,2,1)# BLN D
        keys=keys.permute(0,3,2,1)
        values=values.permute(0,3,2,1)
        queries=queries.reshape(Bq*Lq,Nq,Dq)
        keys=keys.reshape(Bk*Lk,Nk,Dk)
        values=values.reshape(Bv*Lv,Nv,Dv)
        queries = self.query_projection(queries).reshape(Bq*Lq,Nq,H,-1)# BNLD
        keys = self.key_projection(keys).reshape(Bk*Lk,Nk,H,-1)#
        values = self.value_projection(values).reshape(Bv*Lv,Nv,H,-1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        shapeo = out.shape
        out = out.reshape(shapeo[0], shapeo[1],-1)
        out = out.reshape(Bq, Lq, shapeo[1], -1).permute(0, 2, 1, 3)


        return self.out_projection(out), attn
class AttentionLayer_crossN(nn.Module):
    def __init__(self, attention, d_model, n_heads, q_dmodel,k_dmodel,v_dmodel,d_keys=None,
                 d_values=None):
        super(AttentionLayer_crossN, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(q_dmodel, d_keys * n_heads)#64 8 8
        self.key_projection = nn.Linear(k_dmodel, d_keys * n_heads)
        self.value_projection = nn.Linear(v_dmodel, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads,v_dmodel)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        Bq, Dq, Nq, Lq= queries.shape
        Bk, Dk, Nk, Lk= keys.shape
        Bv, Dv, Nv, Lv = values.shape
        H = self.n_heads
        queries=queries.permute(0,3,2,1)# BLN D
        keys=keys.permute(0,3,2,1)
        values=values.permute(0,3,2,1)
        queries=queries.reshape(Bq*Lq,Nq,Dq)
        keys=keys.reshape(Bk*Lk,Nk,Dk)
        values=values.reshape(Bv*Lv,Nv,Dv)
        queries = self.query_projection(queries).reshape(Bq*Lq,Nq,H,-1)# BNLD
        keys = self.key_projection(keys).reshape(Bk*Lk,Nk,H,-1)#
        values = self.value_projection(values).reshape(Bv*Lv,Nv,H,-1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        shapeo = out.shape
        out = out.reshape(shapeo[0], -1)
        out = out.reshape(Bq, Lq, 1, -1).permute(0, 2, 1, 3)

        return self.out_projection(out), attn
class Geo_encoder(nn.Module):
    def __init__(self, geo_dim):
        super(Geo_encoder, self).__init__()
        self.geo_dim = geo_dim
        self.geo_head_lon = nn.Sequential(
            nn.Linear(1, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.geo_dim)
        )
        self.geo_head_lat = nn.Sequential(
            nn.Linear(1, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.geo_dim)
        )
        self.geo_head = nn.Sequential(
            nn.Linear(4*self.geo_dim, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.geo_dim)
        )

    def forward(self, x):
        lon = x[:, 0:1]  # shape: (N, 1)
        lat = x[:, 1:2]  # shape: (N, 1)

        # 分别编码经度和纬度
        lon_feat = self.geo_head_lon(lon)    # (N, geo_dim)
        lat_feat = self.geo_head_lat(lat)    # (N, geo_dim)
        lon_sin = torch.sin(lon_feat)  # (N, geo_dim)
        lon_cos = torch.cos(lon_feat)  # (N, geo_dim)
        lat_sin = torch.sin(lat_feat)  # (N, geo_dim)
        lat_cos = torch.cos(lat_feat)  # (N, geo_dim)

        # 拼接四个部分: [sin_lon, cos_lon, sin_lat, cos_lat]
        out = torch.cat([lon_sin, lon_cos, lat_sin, lat_cos], dim=-1)  # (N, 4 * geo_dim)
        out=self.geo_head(out)#N D
        return out
class Tem_encoder(nn.Module):
    def __init__(self, time_dim):
        super(Tem_encoder, self).__init__()
        self.time_dim = time_dim
        self.time_head_y = nn.Sequential(
            nn.Linear(1, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.time_dim)
        )
        """self.time_head_m = nn.Sequential(
            nn.Linear(1, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.time_dim)
        )
        """
        self.time_head_h = nn.Sequential(
            nn.Linear(1, 128),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(128, self.time_dim)
        )
        self.time_head = nn.Sequential(
            nn.Linear(4*self.time_dim, 512),  # 输入经纬度(2维)
            nn.ReLU(),
            nn.Linear(512, self.time_dim)
        )

    def forward(self, x):
        h = x[:, :,0:1]  # shape: (N, 1)
        #m = x[:, :,1:2]  # shape: (N, 1)
        y= x[:,:, 1:2]
        # 分别编码经度和纬度
        h_feat = self.time_head_h(h)    # (N, geo_dim)

        #m_feat = self.time_head_m(m)
        y_feat = self.time_head_y(y)
        h_sin=torch.sin(h_feat)# (N, geo_dim)
        h_cos = torch.cos(h_feat)
        #m_sin = torch.sin(m_feat)
        #m_cos = torch.cos(m_feat)
        y_sin = torch.sin(y_feat)
        y_cos = torch.cos(y_feat)
        # 拼接四个部分: [sin_lon, cos_lon, sin_lat, cos_lat]
        out = torch.cat([h_sin,h_cos,y_sin,y_cos], dim=-1)  # (N, 4 * geo_dim)
        out=self.time_head(out)#N D
        return out
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        """寻找邻居节点"""
        self.k = getattr(args, 'k', 4)  # 默认 k=4
        # === 1. 加载固定坐标 ===
        feature_dir = f"../{args.feature}"
        locations= np.load(os.path.join(feature_dir, 'locations.npy'))  # [N, 2]
        locations_e = np.load(os.path.join(feature_dir, 'locations_e.npy'))# [N, 2]
        cpan_flat = locations_e.reshape(-1, 2)
        cpan_flat = np.radians(cpan_flat)
        station_coord = locations.reshape(-1, 2)  # (2,)
        station_coord = np.radians(station_coord)
        nbrs_pan = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', metric='haversine').fit(cpan_flat)
        _, indices_pan = nbrs_pan.kneighbors(station_coord)
        self.register_buffer('neighbor_indices', torch.from_numpy(indices_pan).long())  # [N, k]
        np.savetxt("neighbors.csv", indices_pan, delimiter=",", fmt="%d")
        wandb.save("neighbors.csv")  # ← 关键！
        self.num_nodes=103
        self.in_dim=args.enc_in
        self.d_model=32
        self.d_ff=64
        self.n_heads=8
        self.attention_dim=64
        self.geo_dim=16
        self.time_dim=16
        self.layer_num=1
        self.seq_len=args.seq_len
        self.out_len=args.seq_len
        self.dropout=0.1
        self.feature=args.feature
        self.dropout_ = nn.Dropout(self.dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1,self.d_model))
        self.device=args.device
        self.o_e_tcn=TCN(self.in_dim,self.d_model,kernel_size=3,num_layers=3,dilation=1)
        self.norm=nn.LayerNorm(32)
        self.layernorm_t = nn.ModuleList(
            nn.LayerNorm(32) for _ in range(self.layer_num)
        )
        self.layernorm_n = nn.ModuleList(
            nn.LayerNorm(32) for _ in range(self.layer_num)
        )
        self.geo_enc=Geo_encoder(self.geo_dim)
        self.tem_enc=Tem_encoder(self.time_dim)
        self.space_cross_atten=AttentionLayer_crossN(
                        FullAttention(False, args.factor, attention_dropout=self.dropout,
                                      output_attention=True), self.d_ff, self.n_heads,self.d_model+self.geo_dim,self.d_model+self.geo_dim,self.d_model)
        self.self_attention_layer=nn.ModuleList([
        nn.ModuleDict({
            'temporal': AttentionLayer_T(
                        FullAttention(False, args.factor, attention_dropout=self.dropout,
                                      output_attention=True), self.d_ff, self.n_heads,self.d_model+self.time_dim,self.d_model+self.time_dim,self.d_model),
            'spatial': AttentionLayer_N(
                        FullAttention(False, args.factor, attention_dropout=self.dropout,
                                      output_attention=True), self.d_ff,self.n_heads,self.d_model+self.geo_dim,self.d_model+self.geo_dim,self.d_model)
        })
        for _ in range(self.layer_num)  # 3层
    ])
        self.projection = nn.Linear(
            self.d_model, 1, bias=True)

    def GeoEmbedding(self,geop):
        return self.geo_head(geop)
    def TimeEmbedding(self,datee):
        return self.time_head(datee)

    def sin_cos_position(self,ctra,std_geo):
        # input:16,2
        # 首先对经纬度进行周期标准化，确保经纬度数值范围在(0,2pi)
        ctra=torch.Tensor(ctra).to(self.device)
        std_geo=torch.Tensor(std_geo).to(self.device)
        min_lon, max_lon = std_geo[:, 1].min(), std_geo[:, 1].max()  # 经度范围
        min_lat, max_lat = std_geo[:, 0].min(), std_geo[:, 0].max()  # 纬度范围

        # 归一化到 [0, 1]
        lon_nor = (ctra[:, 1] - min_lon) / (max_lon - min_lon)  -0.5# (lon + 125)/11
        lat_nor = (ctra[:, 0] - min_lat) / (max_lat - min_lat) -0.5# (lat - 32)/11
        pos_enc = torch.stack([lat_nor, lon_nor], dim=-1)
        return pos_enc

    def encode_time_features(self, datee):
            year = datee[:, :, 0]
            year_nor = year - year.min()
            month = datee[:, :, 1]
            month_nor = (month - 1) / 11
            week = datee[:, :, 2]
            week_nor = (week - 0) / 6
            hour = datee[:, :, 3]
            hour_nor = (hour - 0) / 23
            datee_normalized = torch.cat([
                year_nor.unsqueeze(-1),  # [B, T, 1]
                month_nor.unsqueeze(-1),  # [B, T, 1]
                week_nor.unsqueeze(-1),  # [B, T, 1]
                hour_nor.unsqueeze(-1)  # [B, T, 1]
            ], dim=-1)  # 沿着最后一个维度拼接

            return datee_normalized  # 形状为 [B, T, 4]


        # 找到单个站点的周围k个盘古网格
    def find_k_nearest_neighbors(self, pan_fut, ctra, cpan, k):

            """
            找到每个站点 (N) 的 k 个近邻的 pan_fut 数据点。

            参数:
                pan_fut: (B, C, lat, lon, L) 的 PAN 未来数据
                ctra:(1, 2) 的站点坐标 (lat、lon)
                cpan: ndarray, (lat, lon, 2) 的 PAN 网格坐标
                k: int, 要找到的近邻数量

            返回:
                era_k: ndarray, (B, C, N, k, L) 的 ERA 近邻数据
                pan_k: ndarray, (B, C, N, k, L) 的 PAN 近邻数据
            """

            # cpan

            cpan_flat = cpan.reshape(-1, 2)  # (lat * lon, 2)
            cpan_flat = np.radians(cpan_flat)

            # B,C,_,_,L=pan_fut.shape
            # pan_fut = pan_fut.reshape(B, C, -1, L)

            # 初始化最近邻模型
            nbrs_pan = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric='haversine').fit(cpan_flat)
            station_coord = np.array(ctra).reshape(1, 2)  # (2,)
            station_coord = np.radians(station_coord)
            # 获取该站点最近的 k 个PAN 网格点索引
            _, indices_pan = nbrs_pan.kneighbors(station_coord)
            indices_pan = torch.Tensor(indices_pan).to(self.device).long()
            pan_k = pan_fut[:, :, indices_pan, :]  # pan_fut:(B,C,1,k,L)
            cpan_n = torch.Tensor(cpan_flat[indices_pan.cpu(), :]).to(self.device)#1 k 2
            return pan_k.squeeze(2),cpan_n# B,C,k,L

    def forward(self,obs_his, era_his, cobs, cera, mask, datee):

        '''
        BNTC
        BN2TC
        N 2
        N2 2
        B N T C
        B T 4
        '''
        B,N,L,C = obs_his.shape
        obs_his_1=obs_his
        obs_his_2=obs_his
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
        mean_enc_o =torch.where(
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
        '''TCN'''
        obs_his=obs_his.permute(0,3,1,2)#B C N L

        #print('3次之后')
        #print(obs_his[0,0,0,:])
        """obs_t=self.o_e_tcn(obs_his)#B D N L
        #print(obs_t.shape)
        era_t=self.o_e_tcn(era_his)#B D N L"""
        idx = self.neighbor_indices  # [N2, k]
        N2, K = idx.shape[0], idx.shape[1]
        era_his = era_his.permute(1,0,2,3)  # BNTC-N B T C
        neighbors_era=era_his[idx]#N2 K B T C
        N_s,K_e,B_e,L_e,C_e = neighbors_era.shape
        neighbors_era=neighbors_era.reshape( N_s*K_e,B_e,L_e,C_e)#N2*K B L C
        neighbors_era=neighbors_era.permute(1,3,0,2)
        obs_t = self.o_e_tcn(obs_his)  # B D N L
        # print(obs_t.shape)
        era_t = self.o_e_tcn(neighbors_era)#B C N2*K  L
        B, D, N1, L = era_t.shape
        era=era_t.reshape(B,D,N2,K,L)

        era_p = cera[idx]#N K 2
        era_p=era_p.reshape(-1,2)
        era_p = self.sin_cos_position(era_p,cera)
        era_p= self.geo_enc(era_p)# n*k 32
        era_p=era_p.reshape(N,self.k,-1)
        era_p=era_p.unsqueeze(0).unsqueeze(-1)
        era_p=era_p.repeat(B,1,1,1,L).permute(0,3,1,2,4)# B 36 N K L
        era_t_gc = torch.cat([era, era_p], dim=1) #B D+36 N K L

        '''GC'''
        cobs = self.sin_cos_position(cobs,cera)
        cobs_enc= self.geo_enc(cobs)# N 36
        cobs_enc_o = cobs_enc.unsqueeze(0).unsqueeze(-1)
        cobs_enc = cobs_enc_o.repeat(B, 1, 1, L).permute(0, 2, 1, 3)#B 32 N L


        '''TC'''
        date_enc=self.tem_enc(datee)
        date_enc_o = date_enc.unsqueeze(1)
        date_enc = date_enc_o.repeat(1, N, 1, 1).permute(0,3,1,2)#B 32 N L
        '''可学习向量'''
        # 使用 torch.where，只对 mask == 0 的位置加上 mask_vector
        mask_bool=mask.repeat(1,1,1,self.d_model).bool()# B N L D
        new_obs_t=obs_t.permute(0,2,3,1)+ self.mask_token
        new_obs=obs_t.permute(0,2,3,1)
        #print(mask.shape)
        obs_t = torch.where(mask_bool,  new_obs,new_obs_t)
        obs_t=obs_t.permute(0,3,1,2)
        '''单t 空间交叉注意力'''
        obs_t_gc=torch.cat([obs_t,cobs_enc],dim=1)# Q   [B D+16 N L]
        obs_t_gc=obs_t_gc.unsqueeze(3)
        era_t_gc=era_t_gc.permute(0,2,1,3,4)
        obs_t_gc = obs_t_gc.permute(0, 2, 1, 3, 4)
        era_t_gc=era_t_gc.reshape(B*N,-1,self.k,L)
        obs_t_gc=obs_t_gc.reshape(B*N,-1,1,L)
        era_t=era.permute(0,2,1,3,4).reshape(B*N,-1,self.k,L)
        obs_spati,atten_N2=self.space_cross_atten(obs_t_gc,era_t_gc,era_t,attn_mask=None) #BN D 1 L
        obs_spati=obs_spati.squeeze(1).reshape(B,N,L,-1)
        obs_spati=self.dropout_(obs_spati)
        obs_t=obs_t.permute(0,2,3,1)
        #print('单t 空间交叉')
        #print(obs_spati[0,0,0,:])
        '''残差'''
        obs_t=obs_t+obs_spati
        obs_t=self.norm(obs_t)
        #print('残差')
        #print(obs_t[0, 0, 0, :])
        obs=obs_t.permute(0,3,1,2)
        '''时空自注意力层'''
        for i in range(self.layer_num):
            obs_t=torch.cat([obs,date_enc],dim=1)# QK [B D+18 N L]
            obs_t_mid,atten_T = self.self_attention_layer[i]['temporal'](obs_t,obs_t,obs,attn_mask=None)
            res_t=obs.permute(0,2,3,1)
            obs_t_mid=obs_t_mid.reshape(B,N,L,-1)
            obs_t_mid = self.dropout_(obs_t_mid)
            obs_t_mid=obs_t_mid+res_t
            obs_t_mid = self.layernorm_t[i](obs_t_mid)
            #print('单n 时间自')
            #print(obs_t_mid[0, 0, 0, :])
            obs_t_mid=obs_t_mid.permute(0,3,1,2)
            obs_s=torch.cat([obs_t_mid,cobs_enc],dim=1)# QK [B D+12 N L]
            obs,atten_N= self.self_attention_layer[i]['spatial'](obs_s,obs_s,obs_t_mid,attn_mask=None)
            res_n=obs_t_mid.permute(0,2,3,1)
            obs=obs.reshape(B,N,L,-1)
            obs = self.dropout_(obs)
            obs=obs+res_n
            obs = self.layernorm_n[i](obs)
            obs=obs.permute(0,3,1,2)
            #print('单t 空间自')
            #print(obs[0, 0, 0, :])
        obs=obs.permute(0,2,3,1)
        '''FC'''
        out = self.projection(obs)
        out = out.mul(
            (std_enc_o[:, :, 0, :].unsqueeze(2).repeat(
                1, 1, self.out_len, 1)))
        out = out.add(
            (mean_enc_o[:, :, 0, :].unsqueeze(2).repeat(
                1, 1, self.out_len, 1)))
        shapea=atten_N.shape
        atten_N=atten_N.reshape(B,-1,shapea[1],shapea[2],shapea[3])
        return out,atten_N.mean(dim=(1, 2))  #B N N
        #return out, out,atten_T,atten_N,atten_N2




