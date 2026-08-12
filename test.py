import argparse
import torch
import torch.nn as nn
from pathlib import Path

from model import S2WAT
from model import AdaFormer
from net import Decoder_MVGG
from tools import save_transferred_imgs, Sample_Test_Net
from object_detection import find_best_pair


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_path', type=str, default='./input/Test/Content/000000001371.jpg',
                    help='Path to the content image')
parser.add_argument('--style_dir', type=str, default='./input/Test/Style',
                    help='Directory path to style images')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory to save the output image(s)')
parser.add_argument('--checkpoint_import_path', type=str, default='./pre_trained_models/checkpoint/checkpoint_40000_epoch.pkl',
                    help='Directory path to the importing checkpoint')

args = parser.parse_args()

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)
print()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

"""
# Models Config
transModule_config = TransModule_Config(
            nlayer=3,
            d_model_list=[768, 768, 384],
            nhead=8,
            mlp_ratio=4,
            qkv_bias=False,
            attn_drop=0.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            norm_first=True
            )
"""

# Hardware Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Models
encoder = S2WAT(
  img_size=224,
  patch_size=2,
  in_chans=3,
  embed_dim=192,
  depths=[2, 2, 2],
  nhead=[3, 6, 12],
  strip_width=[2, 4, 7],
  drop_path_rate=0.,
  patch_norm=True
)
decoder = Decoder_MVGG(d_model=384)
adaformer = AdaFormer(activation="softmax")

network = Sample_Test_Net(encoder, decoder, adaformer)

# Load the checkpoint
print('loading checkpoint...')
checkpoint = torch.load(args.checkpoint_import_path, map_location=device, weights_only=False)

network.encoder.load_state_dict(checkpoint['encoder'])
network.decoder.load_state_dict(checkpoint['decoder'])
network.AdaFormer.load_state_dict(checkpoint['AdaFormer'])

loss_count_interval = checkpoint['loss_count_interval']
print('loading finished')

# Load the model to device 
network.to(device)

# ===============================================Test===============================================

# Step 1: 使用 simple.py 找出最佳 content/style 配對
content_path, style_path = find_best_pair(
    query_image_path=args.content_path,
    candidate_dir=args.style_dir,
)

if style_path is None:
    raise RuntimeError("沒有找到適合的 style 圖片。")

print(f"Content image: {content_path}")
print(f"Selected style image: {style_path}")

# Step 2: 將這組 content/style 圖片送進風格轉換模型
save_transferred_imgs(
    network,
    content_path,
    style_path,
    args.output_dir,
    device=device,
)