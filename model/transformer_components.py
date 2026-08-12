import torch
import torch.nn as nn
from torch import linalg as LA
from .transformer_tools import DropPath, to_2tuple


class Mlp(nn.Module):
  """MLP as implemented in timm
  """
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    drops = to_2tuple(drop)

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drops[0])
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drops[1])

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class Attention(nn.Module):
  """Self Attention as implemented in timm
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    

    self.to_qkv = nn.Linear(d_model, d_model*3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.size()
    qkv = self.to_qkv(x).reshape(B, N, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


# --- Helper Classes ---
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k):
        return self.softmax(torch.bmm(q, k))

class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        q_norm = LA.vector_norm(q, dim=-1, keepdim=True)
        k_norm = LA.vector_norm(k, dim=1, keepdim=True)
        s = torch.bmm(q, k) / torch.bmm(q_norm, k_norm) + 1
        a = s / s.sum(dim=-1, keepdim=True)
        return a

# --- Core Modules ---

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

    def forward(self, fc, fs, fcs):
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
        A = self.activation(Q, K)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h_c, w_c = fc.size()
        M = M.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)
        S = S.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(fcs) + M


class Reshape(nn.Module):
    """
    實作您的 5 步驟包裝器：(B,N,C) <-> (B,C,H,W)
    """
    def __init__(self, dim):
        super().__init__()
        # 步驟 3 的核心模組
        self.core_adaattn = AdaAttN(qkv_dim=dim, activation="softmax")

    def forward(self, fc_seq, fs_seq, fcs_seq, content_shape, style_shape):
        """
        Args:
            fc_seq: Content (B, N_c, C)
            fs_seq: Style (B, N_s, C)
            fcs_seq: Input to modulate (B, N_c, C)
            content_shape: tuple (H_c, W_c)
            style_shape: tuple (H_s, W_s)
        """
        # 步驟 1: 輸入確認
        B, N_c, C = fc_seq.shape
        _, N_s, _ = fs_seq.shape
        H_c, W_c = content_shape
        H_s, W_s = style_shape

        # 步驟 2: Reshape (Sequence -> Image)
        # transpose(1, 2) 把 C 換到 channel 維度
        fc_img = fc_seq.transpose(1, 2).reshape(B, C, H_c, W_c)
        fs_img = fs_seq.transpose(1, 2).reshape(B, C, H_s, W_s)
        fcs_img = fcs_seq.transpose(1, 2).reshape(B, C, H_c, W_c)

        # 步驟 3: 執行 2D AdaAttN
        out_img = self.core_adaattn(fc_img, fs_img, fcs_img)

        # 步驟 4: Reshape Back (Image -> Sequence)
        # flatten(2) 把 H, W 壓扁變成 N
        out_seq = out_img.flatten(2).transpose(1, 2)

        # 步驟 5: 輸出
        return out_seq


class Attention_Cross(nn.Module):
  """Attention for decoder layer.Some palce may be called "inter attention"
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    
    self.to_q = nn.Linear(d_model, d_model, bias=qkv_bias)
    self.to_kv = nn.Linear(d_model, d_model*2, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
    B, Nx, C = x.size()
    _, Ny, _ = y.size()
    q = self.to_q(x).reshape(B, Nx, self.nhead, C // self.nhead).permute(0, 2, 1, 3)
    kv = self.to_kv(y).reshape(B, Ny, 2, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


class TransformerEncoderLayer(nn.Module):
  """Implemented as vit block in timm
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)

    self.attn = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)    
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x):
    if self.norm_first == True:
      x = x + self.drop_path(self.attn(self.norm1(x)))
      x = x + self.drop_path(self.mlp(self.norm2(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn(x)))
      x = self.norm2(x + self.drop_path(self.mlp(x)))
    return x


"""
class TransformerDecoderLayer(nn.Module):

  #Transformer Decoder Layer

  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)
    
    self.attn1 = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.attn2 = Attention_Cross(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.norm3 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x, y):
    
      #Args:
        #x: output of the former layer
        #y: memery of the encoder layer
    
    if self.norm_first == True:
      x = x + self.drop_path(self.attn1(self.norm1(x)))
      x = x + self.drop_path(self.attn2(self.norm2(x), y))
      x = x + self.drop_path(self.mlp(self.norm3(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn1(x)))
      x = self.norm2(x + self.drop_path(self.attn2(x, y)))
      x = self.norm3(x + self.drop_path(self.mlp(x)))
    return x
"""


class TransformerDecoderLayer(nn.Module):
  """
  Modified to support shape passing and AdaAttN
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0., 
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)
    
    self.attn1 = Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    
    # 修改這裡：使用 AdaAttN_With_Reshape 替換原本的 Attention_Cross
    self.attn2 = Reshape(d_model)
    
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.norm3 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

  def forward(self, x, y, content_shape=None, style_shape=None): # 新增形狀參數
    """
      Args:
        x: output of the former layer (Sequence)
        y: memery of the encoder layer (Sequence)
        content_shape: tuple (H, W)
        style_shape: tuple (H, W)
    """
    # 1. Self Attention (不變)
    if self.norm_first:
      x = x + self.drop_path(self.attn1(self.norm1(x)))
    else:
      x = self.norm1(x + self.drop_path(self.attn1(x)))
    
    # 2. AdaAttN (修改處：傳入形狀並使用新模組)
    assert content_shape is not None and style_shape is not None, "AdaAttN requires shape info!"
    
    if self.norm_first:
      # Pre-Norm: 先 Norm 再傳入 AdaAttN
      # 注意：輸入三個參數 (Content, Style, Input_to_modulate)
      # 這裡我們把 x 既當作 Content Query 也當作被調節的對象
      x_modulated = self.attn2(self.norm2(x), y, self.norm2(x), content_shape, style_shape)
      x = x + self.drop_path(x_modulated)
    else:
      # Post-Norm
      x_modulated = self.attn2(x, y, x, content_shape, style_shape)
      x = self.norm2(x + self.drop_path(x_modulated))

    # 3. MLP (不變)
    if self.norm_first:
      x = x + self.drop_path(self.mlp(self.norm3(x)))
    else:
      x = self.norm3(x + self.drop_path(self.mlp(x)))
      
    return x


# Example Decoder_Layer
# import torch
# model = TransformerDecoderLayer(768, nhead=8, norm_first=True)
# tgt = torch.randn(1, 256, 768)
# memory = torch.randn(1, 196, 768)
# output = model(tgt, memory)
# output.shape