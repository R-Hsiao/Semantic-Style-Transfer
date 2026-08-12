import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from model import TransModule_Config
from model import TransformerDecoderLayer


########################################## DINOv2 ##########################################

class DINO_Semantic_Extractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        """
        初始化 DINOv2 語意提取器 (固定輸入 224x224 專用版本)
        """
        super(DINO_Semantic_Extractor, self).__init__()
        
        print(f"Loading DINOv2 ({model_name}) from torch hub...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.backbone.eval()
        
        # 凍結參數
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def extract_patch_features(self, x):
        """
        輸入嚴格為 (B, 3, 224, 224)，輸出將永遠固定為 16x16 的特徵圖
        """
        B, C, H, W = x.shape
        if H != 224 or W != 224:
            raise ValueError(f"DINO Extractor expects 224x224, but got {H}x{W}")
            
        x_norm = self.normalize(x)
        features_dict = self.backbone.forward_features(x_norm)
        patch_tokens = features_dict['x_norm_patchtokens'] # (B, 256, D)
        
        # 因為是 224x224，所以長寬必定是 16x16 (256 個 patch)
        D = patch_tokens.shape[-1]
        patch_features = patch_tokens.permute(0, 2, 1).reshape(B, D, 16, 16)
        
        return patch_features # (B, 384, 16, 16)

    def compute_aligned_similarity(self, F_c, F_s, target_shape):
        """
        將 16x16 的 DINOv2 特徵，插值對齊到 AdaFormer 指定的形狀 (例如 56x56 或是 28x28)
        """
        # 空間解析度對齊
        F_c_up = F.interpolate(F_c, size=target_shape, mode='bilinear', align_corners=False)
        F_s_up = F.interpolate(F_s, size=target_shape, mode='bilinear', align_corners=False)
        
        # 攤平為序列
        F_c_flat = F_c_up.flatten(2) # (B, D, N_t)
        F_s_flat = F_s_up.flatten(2)
        
        # L2 正規化
        F_c_norm = F.normalize(F_c_flat, p=2, dim=1) 
        F_s_norm = F.normalize(F_s_flat, p=2, dim=1) 
        
        # 計算 Cosine Similarity
        S = torch.bmm(F_c_norm.transpose(1, 2), F_s_norm)
        
        # ⚠️ 這裡我們故意保留最原始的 S (包含負數)，不進行 (S+1)/2 的操作
        # 這將在後面的 AdaFormer 模組中透過動態 Z-score 來處理！
        return S

    def forward(self, i_c, i_s, shapes):
        """
        一次性吐出所有需要的 S 矩陣
        Args:
            shapes: list of tuples, 例如 [(56,56), (28,28), (28,28)] 代表 L1, L2, L3
        """
        F_c = self.extract_patch_features(i_c)
        F_s = self.extract_patch_features(i_s)
        
        S_matrices = []
        for shape in shapes:
            S = self.compute_aligned_similarity(F_c, F_s, shape)
            S_matrices.append(S)
            
        # 回傳一個包含 3 個矩陣的列表: [S_L1, S_L2, S_L3]
        return S_matrices

########################################## VGG & components ##########################################

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


########################################## Transfer Module ##########################################

class SeqUpsample(nn.Module):
    """將 Sequence 轉回 2D Image 進行上採樣與通道降維，再轉回 Sequence"""
    def __init__(self, in_dim, out_dim):
        super(SeqUpsample, self).__init__()
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 使用 1x1 卷積來進行通道降維 (例如 768 -> 384)
        self.reduce_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x_seq, shape, target_shape):
        B, N, C = x_seq.shape
        H, W = shape
        target_H, target_W = target_shape
        
        # 1. Sequence to Image (B, C, H, W)
        x_img = x_seq.transpose(1, 2).reshape(B, C, H, W)
        
        # 2. 上採樣與降維
        x_img = F.interpolate(x_img, size=(target_H, target_W), mode='nearest')
        x_img = self.reduce_conv(x_img)
        
        # 3. Image to Sequence (B, N_new, C_new)
        out_seq = x_img.flatten(2).transpose(1, 2)
        
        return out_seq, (target_H, target_W)

class TransModule(nn.Module):
  """
  Multi-scale Transfer Module of Style Transfer via Transformer
  """
  def __init__(self, config: TransModule_Config=None):
    super(TransModule, self).__init__()
    
    # 1. 根據 d_model_list (例如 [768, 768, 384]) 建立對應維度的 Decoder Layer
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=dim,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for dim in config.d_model_list
    ])
    
    # 2. 宣告中層(768)到淺層(384)的過渡模組
    # 注意：深層(768)到中層(768)解析度與通道皆相同，不需要此模組
    self.up_mid2shal = SeqUpsample(in_dim=768, out_dim=384)

  def forward(self, f_c_list, f_s_list):
    """
    Args:
      f_c_list: 包含淺、中、深層內容特徵的列表。每個元素為 (Tensor, arbitrary_input, (H, W))
      f_s_list: 包含淺、中、深層風格特徵的列表。每個元素為 (Tensor, arbitrary_input, (H, W))
    """
    # 展開特徵 (index 0=淺層, 1=中層, 2=深層)
    c_shallow_seq, _, c_shape_s = f_c_list[0]
    c_middle_seq,  _, c_shape_m = f_c_list[1]
    c_deep_seq,    _, c_shape_d = f_c_list[2]
    
    s_shallow_seq, _, s_shape_s = f_s_list[0]
    s_middle_seq,  _, s_shape_m = f_s_list[1]
    s_deep_seq,    _, s_shape_d = f_s_list[2]

    # --- 階段一：深層 (Deep) 融合 ---
    # 直接輸入深層特徵
    out_deep = self.layers[0](c_deep_seq, s_deep_seq, content_shape=c_shape_d, style_shape=s_shape_d)
    
    # --- 階段二：中層 (Middle) 融合 ---
    # 形狀相同 (768, 28x28)，直接將深層輸出與中層內容相加
    In_middle = out_deep + c_middle_seq
    out_middle = self.layers[1](In_middle, s_middle_seq, content_shape=c_shape_m, style_shape=s_shape_m)
    
    # --- 階段三：淺層 (Shallow) 融合 ---
    # 中層輸出 (768, 28x28) 上採樣並降維至淺層形狀 (384, 56x56)
    out_middle_up, _ = self.up_mid2shal(out_middle, c_shape_m, target_shape=c_shape_s)
    # 將對齊後的中層輸出與淺層內容相加
    In_shallow = out_middle_up + c_shallow_seq
    out_shallow = self.layers[2](In_shallow, s_shallow_seq, content_shape=c_shape_s, style_shape=s_shape_s)
    
    # 回傳最終的融合序列與其形狀
    return out_shallow, c_shape_s

# Example
# import torch
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=768,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# transModule = TransModule(transModule_config)
# tgt = torch.randn(1, 20, 768)
# memory = torch.randn(1, 10, 768)
# print(transModule(tgt, memory).shape)


########################################## Decoder ##########################################

decoder_stem = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class Decoder_MVGG(nn.Module):
  def __init__(self, d_model=384): # 預設 d_model 改為 384
      super(Decoder_MVGG, self).__init__()
      self.d_model = d_model
      self.decoder = nn.Sequential(
        # --- 原本的 Upsample Layer 2 已經被刪除 ---

        # 從原本的 Upsample Layer 3 開始接手 (現在是第一段放大：56 -> 112)
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 128, 3, 1, 0), # 接收 TransModule 傳來的 384 維
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),

        # 原本的 Upsample Layer 4 (現在是第二段放大：112 -> 224)
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),

        # Channel to 3 (輸出彩色圖片)
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
        
  def forward(self, x):
    x = self.decoder(x)  
    return x


"""
class Decoder_MVGG(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(Decoder_MVGG, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      self.decoder = nn.Sequential(
        # Proccess Layer 1        

        # Upsample Layer 2
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 256, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 4
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),

        # Channel to 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
        
        
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
#       H, W = math.ceil(self.img_H//self.patch_size), math.ceil(self.img_W//self.patch_size)
      (H, W) = input_resolution
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
    x = self.decoder(x)  
    return x
"""

"""
class Decoder_MLP(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(Decoder_MLP, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      
      self.decoder = nn.Sequential(
        # --- Upsample Layer 2 (對應原本的 Block 1) ---
        # 移除 ReflectionPad2d(1)
        # kernel_size=1, padding=0 (這就是 MLP)
        nn.Conv2d(int(self.d_model), 256, 1, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        
        # 額外的 MLP 層 (保持原有的深度結構)
        nn.Conv2d(256, 256, 1, 1, 0),
        nn.ReLU(),
        
        nn.Conv2d(256, 256, 1, 1, 0),
        nn.ReLU(),
        
        nn.Conv2d(256, 256, 1, 1, 0),
        nn.ReLU(),

        # --- Upsample Layer 3 (對應原本的 Block 2) ---
        nn.Conv2d(256, 128, 1, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        
        nn.Conv2d(128, 128, 1, 1, 0),
        nn.ReLU(),

        # --- Upsample Layer 4 (對應原本的 Block 3) ---
        nn.Conv2d(128, 64, 1, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        
        nn.Conv2d(64, 64, 1, 1, 0),
        nn.ReLU(),

        # --- Channel to 3 (Output Layer) ---
        nn.Conv2d(64, 3, 1, 1, 0),
      )
        
        
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
      (H, W) = input_resolution
      # 這裡負責把序列 reshape 回圖片 (B, C, H, W)
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
      
    x = self.decoder(x)  
    return x
"""

# Example 1
# import torch
# decoder = Decoder_MVGG(d_model=768, seq_input=True)
# x = torch.randn(1, 3087, 768)
# y = decoder(x, input_resolution=(63, 49))
# print(y.shape)

class AdaAttnForLoss(nn.Module):
  """
  Parameter-free AdaAttN* for local feature/content loss.
  This module corresponds to AdaAttN* in the paper formula.
  """
  def __init__(self, v_dim, qk_dim, activation="softmax"):
    super(AdaAttnForLoss, self).__init__()
    self.norm_q = nn.InstanceNorm2d(qk_dim, affine=False)
    self.norm_k = nn.InstanceNorm2d(qk_dim, affine=False)
    self.norm_v = nn.InstanceNorm2d(v_dim, affine=False)

    if activation != "softmax":
      raise ValueError("Only softmax activation is supported in this integrated version.")

    self.softmax = nn.Softmax(dim=-1)

  def forward(self, c_x, s_x, c_1x, s_1x):
    """
    c_x:  content VGG feature at layer x, shape (B, C_x, H_x, W_x)
    s_x:  style VGG feature at layer x, shape (B, C_x, H_x, W_x)
    c_1x: concatenated content VGG features from 1 to x
    s_1x: concatenated style VGG features from 1 to x
    """

    # Q from F_c^{1:x}
    Q = self.norm_q(c_1x)
    B, _, H, W = Q.size()
    Q = Q.view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C_1x)

    # K from F_s^{1:x}
    K = self.norm_k(s_1x)
    B, _, H, W = K.size()
    K = K.view(B, -1, H * W)                   # (B, C_1x, N)

    # V from F_s^x
    V = s_x
    B, _, H, W = V.size()
    V = V.view(B, -1, H * W).permute(0, 2, 1)  # (B, N, C_x)

    # Attention
    A = self.softmax(torch.bmm(Q, K))           # (B, N, N)

    # Weighted style mean
    M = torch.bmm(A, V)                         # (B, N, C_x)

    # Weighted style standard deviation
    Var = torch.bmm(A, V ** 2) - M ** 2
    S = torch.sqrt(Var.clamp(min=1e-6))         # (B, N, C_x)

    # Reshape back to feature map
    B, _, Hc, Wc = c_x.size()
    M = M.view(B, Hc, Wc, -1).permute(0, 3, 1, 2)
    S = S.view(B, Hc, Wc, -1).permute(0, 3, 1, 2)

    # AdaAttN* target
    return S * self.norm_v(c_x) + M

########################################## Net ##########################################

class Net(nn.Module):
  def __init__(self, encoder, decoder, AdaFormer, lossNet):
    super(Net, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.encoder = encoder
    self.decoder = decoder
    self.AdaFormer = AdaFormer
    #實例化 DINO 提取器
    self.dino_extractor = DINO_Semantic_Extractor()

    # features of intermediate layers
    lossNet_layers = list(lossNet.children())
    self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
    self.feat_2 = nn.Sequential(*lossNet_layers[4:11]) # relu1_1 -> relu2_1
    self.feat_3 = nn.Sequential(*lossNet_layers[11:18]) # relu2_1 -> relu3_1
    self.feat_4 = nn.Sequential(*lossNet_layers[18:31]) # relu3_1 -> relu4_1
    self.feat_5 = nn.Sequential(*lossNet_layers[31:44]) # relu3_1 -> relu4_1

    # fix parameters
    for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
      for param in getattr(self, name).parameters():
        param.requires_grad = False
    
    # AdaAttN* modules for local feature/content loss
    # x = 3: v_dim = 256, qk_dim = 64 + 128 + 256
    # x = 4: v_dim = 512, qk_dim = 64 + 128 + 256 + 512
    # x = 5: v_dim = 512, qk_dim = 64 + 128 + 256 + 512 + 512
    self.adaattn_for_loss = nn.ModuleList([
      AdaAttnForLoss(256, 64 + 128 + 256, activation="softmax"),
      AdaAttnForLoss(512, 64 + 128 + 256 + 512, activation="softmax"),
      AdaAttnForLoss(512, 64 + 128 + 256 + 512 + 512, activation="softmax"),
    ])

    self.adaattn_for_loss.eval()


  # get intermediate features
  def get_interal_feature(self, input):
    result = []
    for i in range(5):
      input = getattr(self, 'feat_{:d}'.format(i+1))(input)
      result.append(input)
    return result
  
  def feature_down_sample(self, feat_list, last_layer):
    """
    Build F^{1:x} by resizing VGG features from layer 1 to x
    to the spatial size of layer x and concatenating them along channels.

    feat_list:
      index 0 -> relu1_1
      index 1 -> relu2_1
      index 2 -> relu3_1
      index 3 -> relu4_1
      index 4 -> relu5_1

    last_layer:
      3, 4, or 5
    """
    target = feat_list[last_layer - 1]
    size = target.shape[-2:]

    result = []
    for i in range(last_layer - 1):
      down = F.interpolate(
        feat_list[i],
        size=size,
        mode="bilinear",
        align_corners=False
      )
      result.append(down)

    result.append(target)
    return torch.cat(result, dim=1)
  
  def calc_content_loss(self, f_c_loss, f_s_loss, f_i_cs_loss):
    """
    Local feature/content loss corresponding to:

    L_c = sum_{x=3}^{5} || phi_x(I_cs)
          - AdaAttN*(F_c^x, F_s^x, F_c^{1:x}, F_s^{1:x}) ||_2^2
    """
    loss = 0

    for idx, layer in enumerate([3, 4, 5]):
      c_1x = self.feature_down_sample(f_c_loss, layer)
      s_1x = self.feature_down_sample(f_s_loss, layer)

      with torch.no_grad():
        target = self.adaattn_for_loss[idx](
          f_c_loss[layer - 1],
          f_s_loss[layer - 1],
          c_1x,
          s_1x
        )

      loss += self.mse_loss(f_i_cs_loss[layer - 1], target)

    return loss


  def calc_style_loss(self, input, target):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return self.mse_loss(input_mean, target_mean) + \
        self.mse_loss(input_std, target_std)


  # calculate losses
  def forward(self, i_c, i_s):
    # 取得包含淺、中、深的特徵列表 (List)
    f_c_list = self.encoder(i_c)
    f_s_list = self.encoder(i_s)

    # 計算S矩陣
    shapes = [(f.shape[2], f.shape[3]) for f in f_c_list]
    S_matrices = self.dino_extractor(i_c, i_s, shapes)
    lambda_vals = [5.0, 7.0, 7.0]
    
    # 直接將整個 List 餵給AdaFormer
    f_cs = self.AdaFormer(f_c_list, f_s_list, S_matrices=S_matrices, lambda_vals=lambda_vals)
    f_cc = self.AdaFormer(f_c_list, f_c_list)
    f_ss = self.AdaFormer(f_s_list, f_s_list)
    
    # 將結果送給解碼器
    i_cs = self.decoder(f_cs)
    i_cc = self.decoder(f_cc)
    i_ss = self.decoder(f_ss)
    """
    f_c = self.encoder(i_c)
    f_s = self.encoder(i_s)
    f_c_seq, f_c_reso = f_c[0], f_c[2]
    f_s_seq, f_s_reso = f_s[0], f_s[2]
    
    f_cs = self.transModule(f_c_seq, f_s_seq, content_shape=f_c_reso, style_shape=f_s_reso)
    f_cc = self.transModule(f_c_seq, f_c_seq, content_shape=f_c_reso, style_shape=f_c_reso)
    f_ss = self.transModule(f_s_seq, f_s_seq, content_shape=f_s_reso, style_shape=f_s_reso)
    
    i_cs = self.decoder(f_cs, f_c_reso)
    i_cc = self.decoder(f_cc, f_c_reso)
    i_ss = self.decoder(f_ss, f_c_reso)
    """
    
    f_c_loss = self.get_interal_feature(i_c)
    f_s_loss = self.get_interal_feature(i_s)
    f_i_cs_loss = self.get_interal_feature(i_cs)
    f_i_cc_loss = self.get_interal_feature(i_cc)
    f_i_ss_loss = self.get_interal_feature(i_ss)

    loss_id_1 = self.mse_loss(i_cc, i_c) + self.mse_loss(i_ss, i_s)

    loss_c, loss_s, loss_id_2 = 0, 0, 0

    # Local feature loss / paper-style content loss
    loss_c = self.calc_content_loss(f_c_loss, f_s_loss, f_i_cs_loss)

    for i in range(5):
      loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
      loss_id_2 += self.mse_loss(f_i_cc_loss[i], f_c_loss[i]) + self.mse_loss(f_i_ss_loss[i], f_s_loss[i])
    
    return loss_c, loss_s, loss_id_1, loss_id_2, i_cs


# Example 1
# import torch
# from model.s2wat import S2WAT
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=384,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# encoder = S2WAT(
#   img_size=224,
#   patch_size=2,
#   in_chans=3,
#   embed_dim=96,
#   depths=[2, 2, 2],
#   nhead=[3, 6, 12],
#   strip_width=[2, 4, 7],
#   drop_path_rate=0.,
#   patch_norm=True
# )
# transModule = TransModule(transModule_config)
# decoder = Decoder_MVGG(d_model=384, seq_input=True)
# vgg.load_state_dict(torch.load('../input/vggpretrainedmodel/vgg_normalised.pth'))
# net = Net(encoder, decoder, transModule, vgg)
# i_c = torch.randn(1, 3, 224, 224)
# i_s = torch.randn(1, 3, 224, 224)
# loss_c, loss_s, loss_id_1, loss_id_2, i_cs = net(i_c, i_s)
# print(loss_c.item(), loss_s.item(), loss_id_1.item(), loss_id_2.item())
# print(i_cs.shape)