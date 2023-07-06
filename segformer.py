import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.SlWinSegformer import SlWinSegformer
from nets.HR_UVFormer import HR_UVFormer
from nets.MixerSegformer import MixerSegformer
from nets.SwinSegformer import SwinSegformer
from nets.segformer import SegFormer
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#-----------------------------------------------------------------------------------#
# There are 3 parameters that need to be modified to predict using your own trained model
# model_path, backbone and num_classes and all need to be modified
# window_size may need to be modified
#-----------------------------------------------------------------------------------#
class Model(object):

    #---------------------------------------------------#
    # Initialize the Model
    #---------------------------------------------------#
    def __init__(self, model_path, model_type, num_classes_seg, num_classes_cls, phi, input_shape, patch_size, mix_type = 0, cuda = True,
                 classification=False, segmentation=False):
        self.model_path = model_path
        self.model_type = model_type
        self.num_classes_cls = num_classes_cls
        self.num_classes_seg = num_classes_seg
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.mix_type = mix_type
        self.cuda = cuda
        self.phi = phi
        self.classification = classification
        self.segmentation = segmentation
        #---------------------------------------------------#
        # Set different colors for the frame
        #---------------------------------------------------#
        if self.num_classes_seg <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes_seg, 1., 1.) for x in range(self.num_classes_seg)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #    Get the model
        #---------------------------------------------------#
        self.generate()


    def generate(self, onnx=False):
        #-------------------------------#
        #   Loading models and weights
        #-------------------------------#
        if self.model_type == 'Segformer':
            self.net = SegFormer(num_classes_seg=self.num_classes_seg, num_classes_cls=self.num_classes_cls, phi=self.phi,
                              classification=True, segmentation=True)
        elif self.model_type == 'MixerSegformer':
            backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=self.phi,
                                 classification=True, segmentation=True)
            self.net = MixerSegformer(backbone, input_shape=self.input_shape, patch_size=self.patch_size, num_classes=self.num_classes_cls,
                                   num_blocks=4, token_hidden_dim=256,
                                   channel_hidden_dim=2048)
        elif self.model_type == 'SwinSegformer':
            backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=self.phi,
                                 classification=True, segmentation=True)
            self.net = SwinSegformer(backbone, input_shape=self.input_shape,patch_size=200, num_classes=self.num_classes_cls)
        elif self.model_type == 'SlWinSegformer':
            backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=self.phi,
                                 classification=True, segmentation=True, input_shape=self.patch_size)
            self.net = SlWinSegformer(backbone, input_shape=self.input_shape, window_size = 5, patch_size=self.patch_size,
                                   num_classes=self.num_classes_cls)
        elif self.model_type == 'HR_UVFormer':
            backbone_image = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=self.phi, classification=True,
                                       segmentation=True,input_shape=self.patch_size)
            backbone_build = SegFormer(num_classes_seg=2, num_classes_cls=3, phi='b0', classification=True,
                                       segmentation=False, input_shape=self.patch_size, input_channels=3)
            self.net = HR_UVFormer(backbone_image, input_shape=self.input_shape, window_size = 5, num_classes=self.num_classes_cls,
                                building_bone=backbone_build)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))

        self.net.eval() # must eval mode
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                # self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, evaluate_type, input_shape, tif = None,count=False, name_classes=None):
        #---------------------------------------------------------#
        # Convert images to RGB images here to prevent grayscale maps from reporting errors when predicting.
        # The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        # Make a backup of the input image, which is used later for drawing
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]


        image_data, nw, nh  = resize_image(image, (self.input_shape,self.input_shape))

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            with torch.no_grad():

                if tif is None:
                    if self.cuda:
                        images = images.cuda()
                    outputs = self.net(images)
                else:
                    tif = torch.from_numpy(tif)
                    if self.cuda:
                        images = images.cuda()
                        tif = tif.cuda()
                    outputs = self.net(images, tif)

            if evaluate_type == 'segmentation':
                if input_shape != 2000:

                    pr = outputs[1][0]

                    pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()

                    pr = pr[int((self.input_shape - nh) // 2) : int((self.input_shape - nh) // 2 + nh), \
                            int((self.input_shape - nw) // 2) : int((self.input_shape - nw) // 2 + nw)]

                    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
                    #---------------------------------------------------#

                    pr = pr.argmax(axis=-1)
                    result = Image.fromarray(np.uint8(pr))
                else:

                    patch_size = self.patch_size
                    _, rows, cols = np.array(images[0].shape) // patch_size
                    patches = []
                    for row in range(rows):
                        for col in range(cols):
                            patch_image = images[:, :, row * patch_size: (row + 1) * patch_size,
                                          col * patch_size: (col + 1) * patch_size]  # (1,C,H,W)
                            logit_seg = self.net.backbone(patch_image)[1]  # (num_classes,patch_size,patch_size)
                            logit_seg = F.softmax(logit_seg.squeeze(0).permute(1, 2, 0),
                                                  dim=-1).cpu().argmax(axis=-1)  # (patch_size,patch_size)

                            patches.append(logit_seg.unsqueeze(0))
                    pr = torch.cat(patches, dim=0)  # n_patches*(1,H,W) => (n_patches,H,W)
                    # (n_patches,H,W) => (input_shape, input_shape)
                    pr = pr.view(rows, cols, patch_size, patch_size).transpose(1, 2).contiguous().view(input_shape,
                                                                                                       input_shape)

                result = Image.fromarray(np.uint8(pr))

            elif evaluate_type == 'cls_replace_seg':
                if input_shape != 2000:
                    logits_class = outputs[0][0]  # (num_classes)
                    pr = F.softmax(logits_class, dim=-1).cpu().numpy()  # (num_classes)
                    label_num = pr.argmax(axis=-1)  # (class_num)
                    pr = np.full([orininal_w, orininal_h], label_num)
                    result = Image.fromarray(np.uint8(np.full([orininal_w, orininal_h], 0 if label_num == 0 else 1)))

                else:
                    logits_class = outputs[1][0]  # (num_classes, H ,W)
                    _, rows, cols = logits_class.shape
                    patch_size = input_shape // rows
                    pr = F.softmax(logits_class.permute(1, 2, 0),
                                   dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes,H,W) => (H,W)

                    patches = []
                    for row in range(rows):
                        for col in range(cols):
                            logit_seg = torch.full([patch_size, patch_size], pr[row, col])
                            patches.append(logit_seg.unsqueeze(0))
                    pr = torch.cat(patches, dim=0)  # n_patches*(1,H,W) => (n_patches,H,W)
                    # (n_patches,H,W) => (input_shape, input_shape)
                    pr = pr.view(rows, cols, patch_size, patch_size).transpose(1, 2).contiguous().view(input_shape,
                                                                                                       input_shape)

                    result = pr.clone()
                    # result[result >= 1] = 1
                    result = Image.fromarray(np.uint8(result))


            elif evaluate_type == 'cls_to_seg':
                if input_shape != 2000:
                    logits_class = outputs[0][0]  # (num_classes)
                    pr = F.softmax(logits_class, dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes)
                    if pr == 2:
                        logits_seg = outputs[1][0]  # (num_classes,H,W)
                        pr = F.softmax(logits_seg.permute(1, 2, 0),
                                       dim=-1).cpu().numpy()  # (num_classes,H,W) => (H,W,num_classes)

                        pr = pr[int((self.input_shape - nh) // 2): int((self.input_shape - nh) // 2 + nh), \
                             int((self.input_shape - nw) // 2): int((self.input_shape - nw) // 2 + nw)]

                        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

                        pr = pr.argmax(axis=-1)
                    else:
                        pr = np.full([orininal_w, orininal_h], pr)

                else:
                    logits_class = outputs[1][0]  # (num_classes, H ,W)
                    _, rows, cols = logits_class.shape
                    patch_size = input_shape // rows
                    pr = F.softmax(logits_class.permute(1, 2, 0),
                                   dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes,H,W) => (H,W)
                    a = np.array(pr)
                    patches = []
                    for row in range(rows):
                        for col in range(cols):
                            if pr[row, col] == 2:
                                patch_image = images[:, :, row * patch_size: (row + 1) * patch_size,
                                              col * patch_size: (col + 1) * patch_size]  # (1,C,H,W)
                                logit_seg = self.net.backbone(patch_image)[1]  # (num_classes,H,W)
                                logit_seg = F.softmax(logit_seg.squeeze(0).permute(1, 2, 0),
                                                      dim=-1).cpu().argmax(axis=-1) # (patch_size,patch_size)
                            else:
                                logit_seg = torch.full([patch_size, patch_size], pr[row, col])
                            patches.append(logit_seg.unsqueeze(0))
                    pr = torch.cat(patches, dim=0)  # n_patches*(1,H,W) => (n_patches,H,W)
                    # (n_patches,H,W) => (input_shape, input_shape)
                    pr = pr.view(rows, cols, patch_size, patch_size).transpose(1, 2).contiguous().view(input_shape,
                                                                                                       input_shape)

                result = Image.fromarray(np.uint8(pr))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes_seg])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes_seg):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   Blend the new image with the original image and
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.3)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))
        
        return image, result

    def get_miou_png(self, image, evaluate_type, input_shape):

        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]

        image_data, nw, nh  = resize_image(image, (self.input_shape,self.input_shape))

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            if evaluate_type == 'segmentation':

                pr = self.net(images)[1][0]

                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()

                pr = pr[int((self.input_shape - nh) // 2) : int((self.input_shape - nh) // 2 + nh), \
                        int((self.input_shape - nw) // 2) : int((self.input_shape - nw) // 2 + nw)]

                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)

                pr = pr.argmax(axis=-1)
            elif evaluate_type == 'cls_replace_seg':
                logits_class = self.net(images)[0][0]  # (num_classes)
                pr = F.softmax(logits_class, dim=-1).cpu().numpy()  # (num_classes)
                label_num = pr.argmax(axis=-1)  # (class_num)

                if label_num >= 1:  # For patches classified as 1 or 2, pixel labels are all assigned as 1
                    label_num = 1
                pr = np.full([orininal_w, orininal_h], label_num)
            elif evaluate_type == 'cls_to_seg':
                outputs = self.net(images)
                if input_shape != 2000:
                    logits_class = outputs[0][0]  # (num_classes)
                    pr = F.softmax(logits_class, dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes)
                    if pr == 2:
                        logits_seg = outputs[1][0]  # (num_classes,H,W)
                        pr = F.softmax(logits_seg.permute(1, 2, 0),
                                       dim=-1).cpu().numpy()  # (num_classes,H,W) => (H,W,num_classes)

                        pr = pr[int((self.input_shape - nh) // 2): int((self.input_shape - nh) // 2 + nh), \
                             int((self.input_shape- nw) // 2): int((self.input_shape - nw) // 2 + nw)]

                        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

                        pr = pr.argmax(axis=-1)
                    else:
                        pr = np.full([orininal_w, orininal_h], pr)
                else:
                    logits_class = outputs[1][0]  # (num_classes, H ,W)
                    _, rows, cols = logits_class.shape
                    patch_size = input_shape // rows
                    pr = F.softmax(logits_class.permute(1, 2, 0),
                                   dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes,H,W) => (H,W)
                    patches = []
                    for row in range(rows):
                        for col in range(cols):
                            if pr[row, col] == 2:
                                patch_image = images[:, :, row * patch_size: (row + 1) * patch_size,
                                              col * patch_size: (col + 1) * patch_size]  # (1,C,H,W)
                                logit_seg = self.net.backbone(patch_image)[1]  # (num_classes,H,W)
                                logit_seg = F.softmax(logit_seg.squeeze(0).permute(1, 2, 0),
                                                      dim=-1).cpu().argmax(axis=-1)
                            else:
                                logit_seg = torch.full([patch_size, patch_size], pr[row, col])
                            patches.append(logit_seg.unsqueeze(0))
                    pr = torch.cat(patches, dim=0)  # n_patches*(1,H,W) => (n_patches,H,W)
                    # (n_patches,H,W) => (input_shape, input_shape)
                    pr = pr.view(rows, cols, patch_size, patch_size).transpose(1, 2).contiguous().view(input_shape,
                                                                                                       input_shape)

        image = Image.fromarray(np.uint8(pr))
        return image
