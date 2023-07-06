import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from osgeo import gdal
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor, preprocess_input_bud, preprocess_input_poi


class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, dataset, dataset_type):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.dataset            = dataset
        self.dataset_type       = dataset_type

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, self.dataset,"JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, self.dataset, "SegmentationClass"), name + "_patch.png"))
        if self.dataset_type == 'RSB':
            tif = gdal.Open(os.path.join(os.path.join(self.dataset_path, self.dataset, "BuldingFeatures"), name + ".tif"))
            tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
            tif = np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1])
        elif self.dataset_type == 'RSP':
            tif = gdal.Open(os.path.join(os.path.join(self.dataset_path, self.dataset, "POIFeatures"), name + ".tif"))
            tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
            tif = np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1])
        else:
            tif     = None
        #-------------------------------#
        #   Data Enhancement
        #-------------------------------#
        if self.input_shape[0] < 1000:
            jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        # Converted to the form of one_hot
        # +1 is needed here because some of the labels in the voc dataset have a white border
        # We need to ignore the white part, +1 is to facilitate the ignoring.
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        if self.input_shape[0] < 1000:
            seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        else:
            seg_labels = seg_labels.reshape((int(png.shape[0]), int(png.shape[1]), self.num_classes + 1))

        return jpg, png, tif, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        # Get the height and width of the image with the target height and width
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        # Scale the image and distort the length and width
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        # Flip the image
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        # Add gray bars to the excess parts of the image
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)
        #------------------------------------------#
        # Gaussian blur
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        # Rotation
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        # Perform color gamut transformations on images
        # Calculate the parameters of the color gamut transformation
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        # Transferring images to HS
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        # Application Transformation
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    tifs        = []
    seg_labels  = []
    for img, png, tif, labels in batch:
        images.append(img)
        pngs.append(png)
        tifs.append(tif)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    if tifs[0] is None:
        tifs = None
    else:
        tifs = torch.from_numpy(np.array(tifs)).type(torch.FloatTensor)
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, tifs, seg_labels



class ClassificationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, dataset):
        super(ClassificationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.dataset            = dataset
        if self.train:
            self.labels = np.array(pd.read_csv(os.path.join(self.dataset_path, self.dataset, 'ImageSets/Classification', 'train_labels.txt'), header=None, sep='\t').iloc[:, 1],
                              dtype=np.int16)
        else:
            self.labels = np.array(pd.read_csv(os.path.join(self.dataset_path, self.dataset, 'ImageSets/Classification', 'val_labels.txt'), header=None, sep='\t').iloc[:, 1],
                              dtype=np.int16)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        # Read images from files
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, self.dataset,"JPEGImages"), name + ".jpg"))
        target      = self.labels[index]
        #-------------------------------#
        # Data Enhancement
        #-------------------------------#
        jpg = self.get_random_data(jpg,  self.input_shape, random=self.train)
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2, 0, 1])

        # -------------------------------------------------------#
        # Converted to the form of one_hot
        # -------------------------------------------------------#
        labels_one_hot = np.eye(self.num_classes)[target]

        return jpg, target, labels_one_hot

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)

        # ------------------------------#
        # Get the height and width of the image with the target height and width
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            return new_image

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)


        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        image_data = np.array(image, np.uint8)

        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data


def class_dataset_collate(batch):
    images = []
    tgts = []
    labels_one_hot = []
    for img, png, labels in batch:
        images.append(img)
        tgts.append(png)
        labels_one_hot.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(tgts)).long()
    labels_one_hot = torch.from_numpy(np.array(labels_one_hot)).type(torch.FloatTensor)
    return images, pngs, labels_one_hot

class ClsAndSegDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes_cls, num_classes_seg, train, dataset_path, dataset, dataset_type):
        super(ClsAndSegDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes_cls    = num_classes_cls
        self.num_classes_seg    = num_classes_seg
        self.train              = train
        self.dataset_path       = dataset_path
        self.dataset            = dataset
        self.dataset_type = dataset_type

        if self.train:
            self.labels = np.array(pd.read_csv(os.path.join(self.dataset_path, self.dataset, 'ImageSets', 'train_labels.txt'), header=None, sep='\t').iloc[:, 1],
                              dtype=np.int16)
        else:
            self.labels = np.array(pd.read_csv(os.path.join(self.dataset_path, self.dataset, 'ImageSets', 'val_labels.txt'), header=None, sep='\t').iloc[:, 1],
                              dtype=np.int16)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, self.dataset,"JPEGImages"), name + ".jpg"))
        if self.dataset_type == 'RSB':
            tif = gdal.Open(os.path.join(os.path.join(self.dataset_path, self.dataset, "BuldingFeatures"), name + ".tif"))
            tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
            tif = np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1])
        elif self.dataset_type == 'RSP':
            tif = gdal.Open(os.path.join(os.path.join(self.dataset_path, self.dataset, "POIFeatures"), name + ".tif"))
            tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
            tif = np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1])
        else:
            tif     = None
        target_cls      = self.labels[index]
        target_seg = Image.open(
            os.path.join(os.path.join(self.dataset_path, self.dataset, "SegmentationClass"), name + ".png"))


        jpg, target_seg = self.get_random_data(jpg, target_seg, self.input_shape, random=self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float32)), [2,0,1])
        target_seg         = np.array(target_seg)
        target_seg[target_seg >= self.num_classes_seg] = self.num_classes_seg

        seg_labels          = np.eye(self.num_classes_seg + 1)[target_seg.reshape([-1])]
        labels_one_hot_seg  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes_seg + 1))

        labels_one_hot_cls = np.eye(self.num_classes_cls)[target_cls]


        return jpg, target_cls, target_seg, labels_one_hot_cls, labels_one_hot_seg, tif

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))


        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label

def cls_seg_dataset_collate(batch):
    images = []
    tgts_cls = []
    tgts_seg = []
    labels_one_hot_cls = []
    labels_one_hot_seg = []
    tifs = []
    for img, target_cls, target_seg, label_one_hot_cls, label_one_hot_seg, tif in batch:
        images.append(img)
        tgts_cls.append(target_cls)
        tgts_seg.append(target_seg)
        tifs.append(tif)
        labels_one_hot_cls.append(label_one_hot_cls)
        labels_one_hot_seg.append(label_one_hot_seg)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    tgts_cls = torch.from_numpy(np.array(tgts_cls)).long()
    labels_one_hot_cls = torch.from_numpy(np.array(labels_one_hot_cls)).type(torch.FloatTensor)
    if tifs[0] is None:
        tifs = None
    else:
        tifs = torch.from_numpy(np.array(tifs)).type(torch.FloatTensor)

    tgts_seg        = torch.from_numpy(np.array(tgts_seg)).long()
    labels_one_hot_seg  = torch.from_numpy(np.array(labels_one_hot_seg)).type(torch.FloatTensor)
    return images, tgts_cls, tgts_seg, labels_one_hot_cls, labels_one_hot_seg, tifs


def collate_fn(classification, segmentation):
    if segmentation and not classification:
        return seg_dataset_collate
    elif classification and not segmentation:
        return class_dataset_collate
    else:
        return cls_seg_dataset_collate