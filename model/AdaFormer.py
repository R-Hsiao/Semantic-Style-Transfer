import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA


# Activation Functions With Semantic Similarity
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, S_dino=None, lambda_val=0.0):
        """
        計算加入語意先驗的注意力矩陣。
        Args:
            q: 內容 Query 特徵，形狀 (B, N_c, C)
            k: 風格 Key 特徵，形狀 (B, C, N_s)
            S_dino: 由 DINOv2 計算的語意相似度矩陣，形狀 (B, N_c, N_s)
            lambda_val: 語意約束的強度係數 (大於 0 時才會啟動語意引導)
        """
        # 1. 計算原始的 Attention Logits
        logits = torch.bmm(q, k)
        
        # 2. 注入語意先驗
        if S_dino is not None and lambda_val > 0.0:
            if logits.shape != S_dino.shape:
                raise ValueError(f"Attention Shape mismatch! Logits: {logits.shape}, S_dino: {S_dino.shape}")
            
            q_mean = logits.mean()
            q_std = logits.std()
            logits_norm = (logits - q_mean) / (q_std + 1e-5)
            # Z-score Standardization
            s_mean = S_dino.mean()
            s_std = S_dino.std()
            
            # 自動平移並拉伸
            S_dynamic = (S_dino - s_mean) / (s_std + 1e-5)
            
            logits = logits_norm + lambda_val * S_dynamic
            
        return self.softmax(logits)

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        q_norm = LA.vector_norm(q, dim=-1, keepdim=True)
        k_norm = LA.vector_norm(k, dim=1, keepdim=True)
        s = torch.bmm(q, k) / torch.bmm(q_norm, k_norm) + 1
        a = s / s.sum(dim=-1, keepdim=True)
        return a

# AdaAttN Core
class AdaAttN(nn.Module):
    def __init__(self, qkv_dim, activation="softmax"):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.f = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.g = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.h = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.norm_q = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(qkv_dim, affine=False)
        
        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, fc, fs, fcs, S_dino=None, lambda_val=0.0):
        # Runtime Shape Check
        if fc.shape[1] != self.qkv_dim:
            raise ValueError(f"AdaAttN Channel Mismatch: Expecting {self.qkv_dim}, got {fc.shape[1]}")

        # Q^T
        Q = self.f(self.norm_q(fc))
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.g(self.norm_k(fs))
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = self.h(fs)
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.activation(Q, K, S_dino=S_dino, lambda_val=lambda_val)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h_c, w_c = fc.size()
        M = M.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)
        S = S.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(fcs) + M

# Cross-layer Transition
class FeatureTransition21(nn.Module):
    """
    (B, 768, 28, 28) -> (B, 384, 56, 56)
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(768, 384, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return x

# AdaFormer Module
class AdaFormer(nn.Module):
    def __init__(self, activation="softmax"):
        super().__init__()

        # L3
        self.adaattn_l3 = AdaAttN(768, activation)

        # L2
        self.adaattn_l2 = AdaAttN(768, activation)

        # transition
        self.transition_21 = FeatureTransition21()

        # L1
        self.adaattn_l1 = AdaAttN(384, activation)

    def forward(self, fc, fs, S_matrices=None, lambda_vals=[0.0, 0.0, 0.0]):
        """
        fc, fs:
            [L1, L2, L3]
        """

        fc_1, fc_2, fc_3 = fc
        fs_1, fs_2, fs_3 = fs

        # 解包 S 矩陣
        if S_matrices is not None:
            S_l1, S_l2, S_l3 = S_matrices
        else:
            S_l1 = S_l2 = S_l3 = None

        # L3
        fcs_3 = self.adaattn_l3(fc_3, fs_3, fc_3, S_dino=S_l3, lambda_val=lambda_vals[2])

        # L2
        fcs_2 = self.adaattn_l2(fc_2, fs_2, fcs_3, S_dino=S_l2, lambda_val=lambda_vals[1])

        # transition
        fcs_2_to_1 = self.transition_21(fcs_2)

        # L1
        fcs_1 = self.adaattn_l1(fc_1, fs_1, fcs_2_to_1, S_dino=S_l1, lambda_val=lambda_vals[0])

        return fcs_1