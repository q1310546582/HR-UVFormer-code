import os

import torch
import pandas as pd
import cv2
import torch.nn.functional as F
import numpy as np
from PIL import Image
from osgeo import gdal
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image, preprocess_input_bud, \
    preprocess_input_poi
from utils.utils_metrics import compute_mIoU, compute_OA_Kappa


def get_pred(net, image, input_shape, cuda, evaluate_type, tif=None):
    # ---------------------------------------------------------#
    # Convert images to RGB images here to prevent grayscale maps from reporting errors when predicting.
    # The code only supports RGB image prediction, all other types of images will be converted to RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    # Add gray bars to the image to achieve undistorted resize
    # You can also resize directly for recognition
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))
    # ---------------------------------------------------------#
    # Add on the batch_size dimension
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
                                0)  # (H,W,C) => (1,C,H,W)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        with torch.no_grad():
            if tif is None:
                if cuda:
                    images = images.cuda()
                #---------------------------------------------------#
                # Images are transmitted to the network for prediction
                #---------------------------------------------------#
                outputs = net(images)
            else:
                tif    = torch.from_numpy(tif)
                if cuda:
                    images = images.cuda()
                    tif    = tif.cuda()
                outputs = net(images, tif)

        # ---------------------------------------------------#
        # Fetch the kind of each pixel point/sample
        # ---------------------------------------------------#
        if evaluate_type == 'classification':
            logits_class = outputs[0][0]  # (num_classes)
            pr = F.softmax(logits_class, dim=-1).cpu().numpy()  # (num_classes)
            return pr.argmax(axis=-1)  # (class_num)
        elif evaluate_type == 'segmentation':
            # Pixel-level evaluation
            logits_seg = outputs[1][0]  # (num_classes,H,W)
            pr = F.softmax(logits_seg.permute(1, 2, 0),
                           dim=-1).cpu().numpy()  # (num_classes,H,W) => (H,W,num_classes)
            # --------------------------------------#
            # Intercept the gray bars off
            # --------------------------------------#
            pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
                 int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            # Perform image resize
            # ---------------------------------------------------#
            if input_shape[0] < 1000: # Avoid resizing the 10*10 prediction output of the hierarchical network to 2000*2000
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            # Take out the kind of each pixel point
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

            image = Image.fromarray(np.uint8(pr))
            return image
        elif evaluate_type == 'cls_replace_seg':
            # Direct pixel-level evaluation using classified patches
            logits_class = outputs[0][0]  # (num_classes)
            pr = F.softmax(logits_class, dim=-1).cpu().numpy()  # (num_classes)
            label_num =  pr.argmax(axis=-1)  # (class_num)
            if label_num >= 1: # For patches classified as 1 or 2, pixel labels are all assigned as 1
                label_num = 1
            label_img =  np.full([orininal_w, orininal_h], label_num)

            image = Image.fromarray(np.uint8(label_img))
            return image
        elif evaluate_type == 'cls_to_seg':
            # Patches classified as 2 continue to be split
            if input_shape[0] < 1000:
                logits_class = outputs[0][0]  # (num_classes)
                pr = F.softmax(logits_class, dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes)
                if pr == 2:
                    logits_seg = outputs[1][0]  # (num_classes,H,W)
                    pr = F.softmax(logits_seg.permute(1, 2, 0),
                                   dim=-1).cpu().numpy()  # (num_classes,H,W) => (H,W,num_classes)

                    pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh), \
                         int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]

                    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

                    pr = pr.argmax(axis=-1)
                else:
                    pr = np.full([orininal_w, orininal_h], pr)
            else:
                logits_class = outputs[1][0]  # (num_classes, H ,W)
                _,rows,cols  = logits_class.shape
                patch_size   = input_shape[0] // rows
                pr = F.softmax(logits_class.permute(1, 2, 0),
                               dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes,H,W) => (H,W)
                patches = []
                for row in range(rows):
                    for col in range(cols):
                        if pr[row,col] == 2:
                            patch_image = images[:, :, row * patch_size: (row + 1) * patch_size,
                                          col * patch_size: (col + 1) * patch_size] # (1,C,H,W)
                            if type(net)  == torch.nn.DataParallel:
                                logit_seg = net.module.backbone(patch_image)[1]  # (num_classes,H,W)
                            else:
                                logit_seg = net.backbone(patch_image)[1]  # (num_classes,H,W)
                            logit_seg = F.softmax(logit_seg.squeeze(0).permute(1, 2, 0),
                                                  dim=-1).cpu().argmax(axis=-1)
                        else:
                            logit_seg = torch.full([patch_size, patch_size], pr[row,col])
                        patches.append(logit_seg.unsqueeze(0))
                pr = torch.cat(patches, dim=0) # n_patches*(1,H,W) => (n_patches,H,W)
                # (n_patches,H,W) => (input_shape, input_shape)
                pr = pr.view(rows,cols,patch_size,patch_size).transpose(1,2).contiguous().view(input_shape[0],input_shape[1])

            image = Image.fromarray(np.uint8(pr))
            return image

def get_metrics(net, input_shape, cuda, num_classes_cls, num_classes_seg, image_ids, dataset_path, dataset, \
             dataset_type = 'RS',out_path=".temp_out", evaluate_type = 'segmentation', name_classes = None):


        image_ids          = [image_id.split()[0] for image_id in image_ids]
        net.eval()

        if evaluate_type == 'segmentation' or evaluate_type == 'cls_replace_seg':
            gt_dir = os.path.join(dataset_path, dataset, "SegmentationClass")
            pred_dir = os.path.join(out_path, dataset, 'detection-results')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(image_ids):

                image_path = os.path.join(dataset_path, dataset, "JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)

                if dataset_type == 'RS':
                    tif = None
                elif dataset_type == 'RSB':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "BuldingFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                                   0)  # (H,W,C) => (1,C,H,W)
                elif dataset_type == 'RSP':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "POIFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                                         0)

                image = get_pred(net, image, input_shape, cuda, evaluate_type, tif)
                save_path = os.path.join(pred_dir, image_id + ".png")
                if not os.path.exists(save_path):
                    save_dir = os.path.join(pred_dir, image_id.split('/')[0])
                    os.makedirs(save_dir, exist_ok=True)
                image.save(save_path)

            print("Calculate miou.")

            if input_shape[0] > 1000:
                label_suffix      = '_patch'
            else:
                label_suffix = ''

            _, _, _, _, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes_seg, name_classes, label_suffix = label_suffix)  # 执行计算mIoU的函数

        elif evaluate_type == 'classification':
            gt_dir = os.path.join(dataset_path, dataset, "ImageSets")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            labels = np.array(pd.read_csv(os.path.join(gt_dir, 'val_labels.txt'), header=None, sep='\t').iloc[:, 1],
                              dtype=np.int16)
            print("Get Overall Accuracy.")
            preds = []
            for image_id in tqdm(image_ids):

                image_path = os.path.join(dataset_path, dataset, "JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                if dataset_type == 'RS':
                    tif = None
                elif dataset_type == 'RSB':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "BuldingFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                                   0)  # (H,W,C) => (1,C,H,W)
                elif dataset_type == 'RSP':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "POIFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                                         0)

                pred = get_pred(net, image, input_shape, cuda, evaluate_type, tif)
                preds.append(pred)
            print("Calculate Overall Accuracy.")
            _, _, _, _, kappa = compute_OA_Kappa(labels, np.array(preds), num_classes_cls, name_classes)  # 执行计算mIoU的函数

        elif evaluate_type == 'cls_to_seg':
            gt_dir = os.path.join(dataset_path, dataset, "SegmentationClass")
            pred_dir = os.path.join(out_path, dataset, 'detection-results')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(image_ids):

                image_path = os.path.join(dataset_path, dataset, "JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)

                if dataset_type == 'RS':
                    tif = None
                elif dataset_type == 'RSB':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "BuldingFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                                   0)  # (H,W,C) => (1,C,H,W)
                elif dataset_type == 'RSP':
                    tif_path = os.path.join(os.path.join(dataset_path, dataset, "POIFeatures"),
                                            image_id + ".tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                    tif = np.expand_dims(np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                                         0)

                image = get_pred(net, image, input_shape, cuda, evaluate_type, tif)
                save_path = os.path.join(pred_dir, image_id + ".png")
                if not os.path.exists(save_path):
                    save_dir = os.path.join(pred_dir, image_id.split('/')[0])
                    os.makedirs(save_dir, exist_ok=True)
                image.save(save_path)

            if input_shape[0] > 1000:
                label_suffix      = '_pixel'
            else:
                label_suffix = ''
            print("Calculate miou.")
            _, _, _, _, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes_seg, name_classes, label_suffix)  # 执行计算mIoU的函数

