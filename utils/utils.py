import os

import numpy as np
import torch
from PIL import Image

from nets.segformer_training import weights_init


#---------------------------------------------------------#
# Model initialization and loading of pre-trained weights
#---------------------------------------------------------#
def model_init_resume(model, model_path, local_rank, pretrained_backbone, device):
    weights_init(model)
    if model_path != '':
        if local_rank <= 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        # Load according to the Key of pre-trained weights and the Key of the model: the same key for weight loading
        # Note: Please set args.pretrained_backbone=True when pre-trained weights are backbone weights
        # Note: When continuing the last training, please set args.pretrained_backbone=False, and provide the completed training parameter file args.pretrained_filename
        # ------------------------------------------------------#
        model_dict = model.state_dict() if pretrained_backbone == False else model.backbone.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict) if pretrained_backbone == False else model.backbone.load_state_dict(
            model_dict)  # Load the weights of the key matching the pre-trained weights
        # ------------------------------------------------------#
        # Show Keys that do not match
        # ------------------------------------------------------#
        if local_rank <= 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


#---------------------------------------------------------#
# Convert images to RGB images to prevent grayscale maps from reporting errors when predicting.
# The code only supports RGB image prediction, all other types of images will be converted to RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
# resize the input image
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

#---------------------------------------------------#
# Acquired Learning Rate
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image -= np.array([123.675, 116.28, 103.53], np.float32)
    image /= np.array([58.395, 57.12, 57.375], np.float32)
    return image

def preprocess_input_bud(image):
    # mean = np.array([1218.3927, 228.2476, 11.708024], dtype=np.float32)
    # std  = np.array([1222.8671, 568.02466, 12.574712], dtype=np.float32)
    mean = np.array([1200.5781, 480.12347, 22.87459], dtype=np.float32)
    std  = np.array([997.5072 , 667.166  , 13.652594], dtype=np.float32)
    image = np.transpose(image, [1, 2, 0])
    # image_std = (image - np.array(image!=0, dtype=np.int8)*mean)/std
    image_std = (image - mean) / std

    return image_std

def preprocess_input_poi(image):
    mean = np.array([292.85516, 1104.7714, 488.35413], dtype=np.float32)
    std  = np.array([485.4687 , 225.20744, 118.7836], dtype=np.float32)
    image = np.transpose(image, [1, 2, 0])
    # image_std = (image - np.array(image!=0, dtype=np.int8)*mean)/std
    image_std = (image - mean) / std

    return image_std

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'b0' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b0_backbone_weights.pth",
        'b1' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b1_backbone_weights.pth",
        'b2' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b2_backbone_weights.pth",
        'b3' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b3_backbone_weights.pth",
        'b4' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b4_backbone_weights.pth",
        'b5' : "https://github.com/bubbliiiing/segformer-pytorch/releases/download/v1.0/segformer_b5_backbone_weights.pth",
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)