
from utils.masking import TriangularCausalMask
from math import sqrt, sin
from utils.complexNN.nn import *
#实现在N上参数共享，在C不参数共享，在L上卷积的TCN
import torch
import torch.nn as nn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True,diag_lambda_init=1.0):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.diag_lambda = nn.Parameter(torch.tensor(diag_lambda_init, dtype=torch.float32))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.diag_lambda > 0 and L == S:
            eye = torch.eye(L, device=queries.device).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            scores = scores + self.diag_lambda * eye

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
class FullAttention_F(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention_F, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout_rate = attention_dropout

    @staticmethod
    def complex_dropout(x, p=0.0, training=True):
        if not training or p == 0.0:
            return x

        # 生成与输入形状相同的随机 mask（实数）
        keep_prob = 1 - p
        mask = torch.empty(x.shape, dtype=torch.float32, device=x.device).uniform_(0, 1) < keep_prob

        # 将 mask 扩展为复数类型（True 变成 1+0j，False 变成 0+0j）
        mask = mask.to(x.dtype)  # 自动转换为 complex64/complex128

        # 缩放并应用
        return (x * mask) / keep_prob

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A_score = complexSoftmax(scale * scores, dim=-1)
        A=self.complex_dropout(A_score, p=self.dropout_rate, training=self.training)
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


class AttentionLayer_crossF(nn.Module):# 64 13 13 13
    def __init__(self, attention, d_model, n_heads, q_dmodel,k_dmodel,v_dmodel,d_keys=None,
                 d_values=None):
        super(AttentionLayer_crossF, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = cLinear(q_dmodel, d_keys * n_heads)#64 8 8
        self.key_projection = cLinear(k_dmodel, d_keys * n_heads)
        self.value_projection = cLinear(v_dmodel, d_values * n_heads)
        self.out_projection = cLinear(d_values * n_heads,v_dmodel)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        Bq, Cq, Nq, Dq= queries.shape
        Bk, Ck, Nk, Dk= keys.shape
        Bv, Cv, Nv, Dv = values.shape
        H = self.n_heads
        queries=queries.reshape(Bq*Cq,Nq,Dq)
        keys=keys.reshape(Bk*Ck,Nk,Dk)
        values=values.reshape(Bv*Cv,Nv,Dv)
        queries = self.query_projection(queries).reshape(Bq*Cq,Nq,H,-1)# BNLD
        keys = self.key_projection(keys).reshape(Bk*Ck,Nk,H,-1)#
        values = self.value_projection(values).reshape(Bv*Cv,Nv,H,-1)

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
        out = out.reshape(Bq, Cq, Nq, -1)

        return self.out_projection(out), attn
import torch.nn.functional as F
class Complex_Attention(nn.Module):
    def __init__(self, args,M,f_dmodel):
        super(Complex_Attention, self).__init__()
        self.M=M
        self.dropout = nn.Dropout(0.1)
        self.attn_mask = None
        self.d_model=f_dmodel
        self.Q = nn.Linear(self.d_model, self.d_model)
        self.K = nn.Linear(self.d_model, self.d_model)
        self.V = nn.Linear(self.d_model, self.d_model)

    def forward(self,xp_emb,xp_temp_emb):
        # attn:Q(B,C,N,D),K,V(B,C,N,M,D)
        B,C,N,_=xp_emb.shape
        q = self.Q(xp_emb.reshape(-1, N, self.d_model))  # BC,N,d
        k = self.K(xp_temp_emb.reshape(-1, N * self.M, self.d_model))  # BC,NM,d
        v = self.V(xp_temp_emb.reshape(-1, N * self.M, self.d_model))  # BC,NM,d
        B, N, D = q.shape  # Queries shape
        _, NM, _ = k.shape  # Keys and Values shape

        scale = 1. / np.sqrt(D)
        scores = torch.einsum("bnd,bmd->bnm", q, k)  # shape: (B, N, NM)

        if self.attn_mask is not None:
            scores.masked_fill_(self.attn_mask == 0, -np.inf)  # -inf to mask out unwanted positions

        A = F.softmax(scale * scores, dim=-1)  # shape: (B, N, NM)
        A = self.dropout(A)
        output = torch.einsum("bnm,bmd->bnd", A, v)  # shape: (B, N, D)
        return output,A
class Frequency_Aware_Delay_Attn(nn.Module):
    def __init__(self, args,M,N):
        super(Frequency_Aware_Delay_Attn, self).__init__()
        self.seq_len=args.seq_len//2+1#13
        self.device=args.device
        self.d_model=64
        self.M=M
        self.N=N
        self.dropout = nn.Dropout(0.1)
        self.attn_mask=None

        M_real=nn.Parameter(torch.randn(N,M, self.seq_len)).to(self.device)
        M_imag=nn.Parameter(torch.randn(N,M, self.seq_len)).to(self.device)
        self.temp=torch.complex(M_real,M_imag)
        self.fc1=nn.Linear(self.seq_len,self.d_model).to(torch.cfloat)
        self.fc2=nn.Linear(self.d_model,self.d_model).to(torch.cfloat)
        self.real_attn=Complex_Attention(args,self.M,self.d_model)
        self.imag_attn=Complex_Attention(args,self.M,self.d_model)
        self.fc3=nn.Linear(self.d_model,self.seq_len).to(torch.cfloat)
        self.predict_layer=nn.Conv2d(1, 1, (1, 1))
        self.layernorm_t=nn.LayerNorm(self.d_model)
        self.dropout_ = nn.Dropout(0.1)

    def forward(self,xp):
        B,C,N,_=xp.shape
        xp=torch.fft.rfft(xp, dim=-1)#B,C,N,L
        xp_self=xp
        temp=self.temp.repeat(B,C,1,1,1)#B,C,N,M,L
        xp_temp=xp.unsqueeze(-2)*temp#B,C,N,M,L
        xp_temp_emb=self.fc2(self.fc1(xp_temp.reshape(-1,self.M,self.seq_len))).reshape(B,C,N,self.M,-1)#B,C,N,M,D

        xp_emb=self.fc2(self.fc1(xp.reshape(-1,self.seq_len))).reshape(B,C,N,-1)#B,C,N,D
        #attn:Q(B,C,N,D),K,V(B,C,N,M,D)
        xp_emb_real=self.dropout_(xp_emb.real)
        xp_emb_imag=self.dropout_(xp_emb.imag)
        xp_temp_emb_real=self.dropout_(xp_temp_emb.real)
        xp_temp_emb_imag=self.dropout_(xp_temp_emb.imag)
        output_real,A_real=self.real_attn(xp_emb_real,xp_temp_emb_real)
        output_imag,A_imag=self.imag_attn(xp_emb_imag,xp_temp_emb_imag)
        output=torch.complex(output_real,output_imag)
        A=torch.complex(A_real,A_imag)
        output=self.fc3(output).reshape(B,C,N,self.seq_len)
        output=torch.fft.irfft(output, dim=-1)
        output=self.predict_layer(output)

        return output, A