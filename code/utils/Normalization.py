import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveNormalizer(nn.Module):
    def __init__(self, d_feature, eps=1e-5, gate_slope=10.0):
        super().__init__()
        self.d_feature = d_feature
        self.eps = eps
        self.gate_slope = gate_slope  # 控制门控的陡峭程度

        # 可学习的 fallback 参数（用于全缺失时）
        self.learned_mean = nn.Parameter(torch.zeros(d_feature))
        self.learned_std = nn.Parameter(torch.ones(d_feature))

    def forward(self, obs_his, era_his, mask):
        """
        obs_his: [B, N, T, C]
        era_his: [B, N, T, C]
        mask:    [B, N, T, C]   1=observed, 0=missing
        """
        B, N, T, C = obs_his.shape

        # -----------------------------
        # Step 1: 计算有效观测数 n
        # -----------------------------
        num_valid=torch.sum(mask == 1, dim=2, keepdim=True)  # [B, N,1, C], 每个变量的有效观测数

        # -----------------------------
        # Step 2: 构造门控权重 alpha ∈ [0,1]
        # -----------------------------
        # 使用 sigmoid 实现平滑过渡：n > 0 时 alpha ≈ 1, n = 0 时 alpha ≈ 0
        alpha = torch.sigmoid(self.gate_slope * (num_valid - 0.5))  # [B, N,1, C]
        # -----------------------------
        # Step 3: 安全计算统计量（即使全缺失也不 nan）
        # -----------------------------
        sum_obs = torch.sum(obs_his * mask, dim=2,keepdim=True)  # [B, N, 1,C]
        mean_stat = sum_obs / num_valid.clamp(min=1)  # 临时除法安全

        # -----------------------------
        # Step 4: 融合统计量与可学习参数
        # -----------------------------
        # 扩展 learned_mean 到 [B, N, 1, C]
        mean_learned = self.learned_mean.view(1, 1, 1, -1).expand_as(mean_stat)

        # 加权融合
        mean_final = alpha * mean_stat + (1 - alpha) * mean_learned  # [B, N, 1, C]

        # -----------------------------
        # Step 5: 去均值
        # -----------------------------
        obs_his_centered = obs_his - mean_final  # [B, N, T, C]

        # -----------------------------
        # Step 6: 标准差的融合（同理）
        # -----------------------------
        sum_sq = torch.sum((obs_his_centered * mask) ** 2, dim=2)  # [B, N, C]
        var_stat = sum_sq / num_valid.clamp(min=1)
        std_stat = torch.sqrt(var_stat + self.eps).unsqueeze(2)  # [B, N, 1, C]

        std_learned = self.learned_std.view(1, 1, 1, -1).expand_as(std_stat)
        std_final = alpha * std_stat + (1 - alpha) * std_learned
        std_final = std_final.clamp(min=1e-6)

        # -----------------------------
        # Step 7: 归一化
        # -----------------------------
        obs_his_normalized = obs_his_centered / std_final

        # -----------------------------
        # Step 8: era_his 归一化（假设完整）
        # -----------------------------
        mean_enc_e = era_his.mean(dim=2, keepdim=True)
        std_enc_e = era_his.std(dim=2, keepdim=True) + self.eps
        era_his_normalized = (era_his - mean_enc_e) / std_enc_e

        return obs_his_normalized, era_his_normalized, mean_final, std_final