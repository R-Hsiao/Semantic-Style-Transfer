import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from net import DINO_Semantic_Extractor

import os
import zipfile
from PIL import Image
from tqdm import tqdm
from datetime import datetime


####################################### Train Tools #######################################

# save the checkpoint
def save_checkpoint(encoder, AdaFormer, decoder, optimizer, scheduler, epoch,
           log_c, log_s, log_id1, log_id2, log_all, loss_count_interval, save_path):
  checkpoint = {
    'encoder': encoder.state_dict() if not encoder is None else None,
    'AdaFormer': AdaFormer.state_dict() if not AdaFormer is None else None,
    'decoder': decoder.state_dict() if not decoder is None else None,
    'optimizer': optimizer.state_dict() if not optimizer is None else None,
    'scheduler': scheduler.state_dict() if not scheduler is None else None,
    'epoch': epoch if not epoch is None else None,
    'log_c': log_c if not log_c is None else None,
    'log_s': log_s if not log_s is None else None,
    'log_id1': log_id1 if not log_id1 is None else None,
    'log_id2': log_id2 if not log_id2 is None else None,
    'log_all': log_all if not log_all is None else None,
    'loss_count_interval': loss_count_interval if not loss_count_interval is None else None
  }

  torch.save(checkpoint, save_path)


######################################## Test Tools #######################################
def showTorchImage(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    mode = transforms.ToPILImage()(image)
    plt.imshow(mode)
    plt.show()
    plt.close()

def zip_dir(zipFile_name, dir_path):
    z = zipfile.ZipFile(zipFile_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            z.write(os.path.join(dirpath, filename))
    z.close()


def open_img_to_pt(img_path, transform=transforms.ToTensor()):
    img = Image.open(img_path)
    img_pt = transform(img).unsqueeze(dim=0)
    return img_pt


def content_style_transTo_pt(i_c_path, i_s_path, i_c_size=None):
    """Resize the pics of arbitrary size to the shape of content image
    """
    i_c_pil = Image.open(i_c_path)
    i_s_pil = Image.open(i_s_path)
    
    if not i_c_size is None:
        i_c_tf = transforms.Compose([
            transforms.Resize(i_c_size),
            transforms.ToTensor()
        ])
    else:
        i_c_tf = transforms.Compose([
            transforms.ToTensor()
        ])
    
    i_s_size = min(i_c_pil.size[1], i_c_pil.size[0])
    i_s_tf = transforms.Compose([
        transforms.Resize(i_s_size),
        transforms.ToTensor()
    ])
    
    i_c_pt = i_c_tf(i_c_pil).unsqueeze(dim=0)
    i_s_pt = i_s_tf(i_s_pil).unsqueeze(dim=0)
    
    return i_c_pt, i_s_pt


@torch.no_grad()
def save_sample_imgs(network, samples_path, img_saved_path, device=torch.device('cpu')):
    """Test and save samples imgs (Fixed Size)
       Args:
           network       : Model that tested
           samples_path  : Path where the samples saved
                           Required two sub-dirs named 'Content' and 'Style'
           img_saved_path: Path to save the results
    """
    sample_dict = {
        '1': [1,2,5,6],
        '2': [3,6,9],
        '3': [4,6,9],
        '4': [1,8,9],
        '5': [1,6,8],
        '6': [1,6,7],
        '7': [1,6,9],
        '8': [1,6,8],
        '9': [1,6,7],
    }
    
    print('Image generation starts:')
    for i_c_num in tqdm(sample_dict.keys()):
        output_imgs = torch.tensor([])
        for i_s_num in sample_dict[i_c_num]:     
            i_c = open_img_to_pt(os.path.join(samples_path, f'Content/{i_c_num}.png')).to(device)
            i_s = open_img_to_pt(os.path.join(samples_path, f'Style/{i_s_num}.png')).to(device)
            i_cs = network(i_c, i_s)
            output_img = torch.cat((i_c.cpu(), i_s.cpu(), i_cs.cpu()), dim=0)
            output_imgs = torch.cat((output_imgs, output_img), dim=0)
        output_name = os.path.join(img_saved_path, f'test_{i_c_num}.png')
        save_image(output_imgs, output_name, nrow=3)


@torch.no_grad()
def save_sample_imgs_arbitrarySize(network, samples_path, img_saved_path, device=torch.device('cpu')):
    """Test and save samples imgs (Arbitrary Size)
       Args:
           network       : Model that tested
           samples_path  : Path where the samples saved
                           Required two sub-dirs named 'Content' and 'Style'
           img_saved_path: Path to save the results
    """
    sample_dict = {
        '1': [1,2,3,4,5,6,7,8,9,10,11],
        '2': [1,2,3,4,5,6,7,8,9,10,11],
    }
    
    print('Image generation starts:')
    for i_c_num in tqdm(sample_dict.keys()):
        output_imgs = []
        i_c_path = os.path.join(samples_path, f'Content/{i_c_num}.jpg')
        for i_s_num in sample_dict[i_c_num]: 
            i_s_path = os.path.join(samples_path, f'Style/{i_s_num}.jpg')
            i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)
            i_cs = network(i_c.to(device), i_s.to(device))
            i_s = transforms.CenterCrop((i_c.shape[2], i_c.shape[3]))(i_s)
            i_cs = transforms.CenterCrop((i_c.shape[2], i_c.shape[3]))(i_cs)
            output_img = torch.cat((i_c.cpu(), i_s.cpu(), i_cs.cpu()), dim=0)
            output_imgs.append(output_img)
            output_imgs = torch.cat(output_imgs, dim=0)
        output_name = os.path.join(img_saved_path, f'test_{i_c_num}.jpg')
        save_image(output_imgs, output_name, nrow=3)


@torch.no_grad()
def save_transferred_imgs(network, content_path, style_path, img_saved_path, device=torch.device('cpu')):
    print('Image generation starts:')

    i_c, i_s = content_style_transTo_pt(content_path, style_path)
    i_cs = network(i_c.to(device), i_s.to(device))

    content_name = os.path.basename(content_path)
    style_name = os.path.basename(style_path)

    stem_c, suffix_c = os.path.splitext(content_name)
    stem_s, suffix_s = os.path.splitext(style_name)

    output_name = os.path.join(img_saved_path, f'{stem_c}_+_{stem_s}{suffix_c}')
    save_image(i_cs, output_name)


@torch.no_grad()
def save_content_leak_imgs(network, samples_path, img_saved_path, rounds=20, device=torch.device('cpu')):
  print('Image generation starts:')

  i_c_names = os.listdir(os.path.join(samples_path, 'Content'))
  i_s_names = os.listdir(os.path.join(samples_path, 'Style'))
  for i_c_name in tqdm(i_c_names):
    for i_s_name in tqdm(i_s_names):
      i_c_path = os.path.join(samples_path, 'Content', i_c_name)
      i_s_path = os.path.join(samples_path, 'Style', i_s_name)
      i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)

      i_c = i_c.to(device)
      i_s = i_s.to(device)
      i_cs = i_c
      for i in range(rounds):
        i_cs = network(i_cs, i_s)

      stem_c, suffix_c = os.path.splitext(i_c_name)
      stem_s, suffix_s = os.path.splitext(i_s_name)
      output_name = os.path.join(img_saved_path, f'{stem_c}_+_{stem_s}.{suffix_c}')
      save_image(i_cs, output_name)


@torch.no_grad()
def caculate_avg_generate_time(network, i_c_path, i_s_path, round=1, device=torch.device('cpu')):
  i_c = open_img_to_pt(i_c_path)
  i_s = open_img_to_pt(i_s_path)
  i_c = i_c.to(device)
  i_s = i_s.to(device)
  
  time_start = datetime.now()
  for i in range(round):
    i_cs = network(i_c, i_s)
  time_end = datetime.now()

  avg_generate_time = ((time_end-time_start).seconds + (time_end-time_start).microseconds/1000000) / round
  return avg_generate_time


@torch.no_grad()
def caculate_avg_generate_time_multiple(network, samples_path, device=torch.device('cpu')):
  i_c_names = os.listdir(os.path.join(samples_path, 'Content'))
  i_s_names = os.listdir(os.path.join(samples_path, 'Style'))

  nums = len(i_c_names) * len(i_s_names)

  time_start = datetime.now()
  for i_c_name in i_c_names:
    for i_s_name in i_s_names:
      i_c_path = os.path.join(samples_path, 'Content', i_c_name)
      i_s_path = os.path.join(samples_path, 'Style', i_s_name)
      i_c, i_s = content_style_transTo_pt(i_c_path, i_s_path)

      i_c = i_c.to(device)
      i_s = i_s.to(device)
      i_cs = network(i_c, i_s)
  time_end = datetime.now()
      
  avg_generate_time = ((time_end-time_start).seconds + (time_end-time_start).microseconds/1000000) / nums
  return avg_generate_time

class Sample_Test_Net(nn.Module):
    def __init__(self, encoder, decoder, AdaFormer):
        super(Sample_Test_Net, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.AdaFormer = AdaFormer
        # 實例化 DINO 提取器
        self.dino_extractor = DINO_Semantic_Extractor()

    def forward(self, i_c, i_s):
        """
        Args:
            i_c: content image, shape (B, 3, H, W)
            i_s: style image, shape (B, 3, H, W)

        Returns:
            i_cs: stylized image
        """

        f_c_list = self.encoder(i_c)
        f_s_list = self.encoder(i_s)

        # 抓取 L1, L2, L3 的空間解析度
        shapes = [(f.shape[2], f.shape[3]) for f in f_c_list]
        
        # 透過DINOv2算出S矩陣
        S_matrices = self.dino_extractor(i_c, i_s, shapes)
        
        # Lambda值
        lambda_vals = [5.0, 7.0, 7.0]

        # AdaFormer: [L1, L2, L3] -> fused shallow feature map
        f_cs = self.AdaFormer(f_c_list, f_s_list, S_matrices=S_matrices, lambda_vals=lambda_vals)

        # Decoder: 2D feature map -> image
        i_cs = self.decoder(f_cs)

        return i_cs

"""
class Sample_Test_Net(nn.Module):
    def __init__(self, encoder, decoder, transModule, patch_size=8):
        super(Sample_Test_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transModule = transModule
        self.patch_size = patch_size

    def forward(self, i_c, i_s, arbitrary_input=False):
        _, _, H, W = i_c.size()
        # 這兩行可以保留，如果有其他地方會調用的話
        self.decoder.img_H = H
        self.decoder.img_W = W
        
        # 1. 取得包含淺、中、深的特徵列表 (List)
        f_c_list = self.encoder(i_c, arbitrary_input)
        f_s_list = self.encoder(i_s, arbitrary_input)
        
        # --- (以下原本手動拆開特徵的兩行 f_c = f_c[0]... 請刪除) ---
        
        # 2. 直接將整個 List 餵給新版 TransModule
        f_cs_seq, f_cs_reso = self.transModule(f_c_list, f_s_list)
        
        # 3. 傳給解碼器還原成圖片
        i_cs = self.decoder(f_cs_seq, f_cs_reso)
        return i_cs
"""

"""
class Sample_Test_Net(nn.Module):
    def __init__(self, encoder, decoder, transModule, patch_size=8):
        super(Sample_Test_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transModule = transModule
        self.patch_size = patch_size

    def forward(self, i_c, i_s, arbitrary_input=False):
        _, _, H, W = i_c.size()
        self.decoder.img_H = H
        self.decoder.img_W = W
        f_c = self.encoder(i_c, arbitrary_input)
        f_s = self.encoder(i_s, arbitrary_input)
        f_c, f_c_reso = f_c[0], f_c[2]
        f_s, f_s_reso = f_s[0], f_s[2]
        f_cs = self.transModule(f_c, f_s, content_shape=f_c_reso, style_shape=f_s_reso)
        i_cs = self.decoder(f_cs, f_c_reso)
        return i_cs
"""