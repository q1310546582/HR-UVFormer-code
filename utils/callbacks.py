import os

import matplotlib
import torch
import torch.nn.functional as F
from osgeo import gdal

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import pandas as pd
import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image, preprocess_input_bud, \
    preprocess_input_poi
from .utils_metrics import compute_mIoU, compute_OA_Kappa


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes_cls, num_classes_seg, image_ids, dataset_path, dataset, log_dir, cuda, \
             out_path=".temp_out", eval_flag=True, period=1, classification = False, segmentation = False):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes_cls    = num_classes_cls
        self.num_classes_seg    = num_classes_seg
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.dataset            = dataset
        self.out_path           = out_path
        self.eval_flag          = eval_flag
        self.period             = period

        self.classification = classification
        self.segmentation = segmentation

        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.OA         = [0]
        self.kappa      = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt" if segmentation == True else "epoch_percision.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_pred(self, image, tif = None):
        #---------------------------------------------------------#
        # Convert images to RGB images here to prevent grayscale maps from reporting errors when predicting.
        # The code only supports RGB image prediction, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        # Add gray bars to the image to achieve undistorted resize
        # You can also resize directly for recognition
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        # Add on the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0) # (H,W,C) => (1,C,H,W)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if tif is None:
                if self.cuda:
                    images = images.cuda()
                #---------------------------------------------------#
                # Images are transmitted to the web for prediction
                #---------------------------------------------------#
                outputs = self.net(images)
            else:
                tif    = torch.from_numpy(tif)
                if self.cuda:
                    images = images.cuda()
                    tif    = tif.cuda()
                outputs = self.net(images, tif)
            #---------------------------------------------------#
            # Fetch the kind of each pixel point/sample
            #---------------------------------------------------#
            if self.classification and not self.segmentation:
                logits_class = outputs[0][0] # (num_classes)
                pr = F.softmax(logits_class, dim=-1).cpu().numpy()  # (num_classes)
                return pr.argmax(axis=-1) # (class_num)
            elif self.segmentation and not self.classification:
                logits_seg =outputs[1][0] # (num_classes,H,W)
                pr = F.softmax(logits_seg.permute(1,2,0),dim = -1).cpu().numpy() # (num_classes,H,W) => (H,W,num_classes)
                # --------------------------------------#
                # Intercept the gray bars off
                # --------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
                # ---------------------------------------------------#
                # Perform image resize
                # ---------------------------------------------------#
                if self.input_shape[0] < 1000:
                    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
                # ---------------------------------------------------#
                # Take out the kind of each pixel point
                # ---------------------------------------------------#
                pr = pr.argmax(axis=-1)
                image = Image.fromarray(np.uint8(pr))
                return image

            elif self.classification and self.segmentation:
                logits_class = outputs[0][0] # (num_classes)
                pr = F.softmax(logits_class, dim=-1).cpu().numpy().argmax(axis=-1)  # (num_classes)
                if pr == 2:
                    logits_seg = outputs[1][0]  # (num_classes,H,W)
                    pr = F.softmax(logits_seg.permute(1, 2, 0),
                                   dim=-1).cpu().numpy()  # (num_classes,H,W) => (H,W,num_classes)

                    pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                         int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

                    pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

                    pr = pr.argmax(axis=-1)
                else:
                    pr = np.full([orininal_w, orininal_h], pr)

                image = Image.fromarray(np.uint8(pr))
                return image
    


    def on_epoch_end(self, epoch, model_eval, dataset_type):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval

            if self.segmentation and not self.classification:
                gt_dir = os.path.join(self.dataset_path, self.dataset, "SegmentationClass")
                pred_dir = os.path.join(self.out_path , self.dataset, 'detection-results')
                if not os.path.exists(self.out_path ):
                    os.makedirs(self.out_path )
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                print("Get miou.")
                for image_id in tqdm(self.image_ids):

                    image_path  = os.path.join(self.dataset_path, self.dataset, "JPEGImages/"+image_id+".jpg")
                    image       = Image.open(image_path)

                    if dataset_type == 'RS':
                        tif = None
                    elif dataset_type == 'RSB':
                        tif_path = os.path.join(os.path.join(self.dataset_path, self.dataset, "BuldingFeatures"),
                                                image_id + ".tif")
                        tif = gdal.Open(tif_path)
                        tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                        tif = np.expand_dims(np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                                             0)  # (H,W,C) => (1,C,H,W)
                    elif dataset_type == 'RSP':
                        tif_path = os.path.join(os.path.join(self.dataset_path, self.dataset, "POIFeatures"),
                                                image_id + ".tif")
                        tif = gdal.Open(tif_path)
                        tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                        tif = np.expand_dims(np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                                             0)

                    image = self.get_pred(image, tif)
                    save_path = os.path.join(pred_dir, image_id + ".png")
                    if not os.path.exists(save_path):
                        save_dir = os.path.join(pred_dir, image_id.split('/')[0])
                        os.makedirs(save_dir, exist_ok=True)
                    image.save(save_path)

                print("Calculate miou.")
                if self.input_shape[0] > 1000:
                    label_suffix = '_patch'
                else:
                    label_suffix = ''
                _, IoUs, _, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes_seg, label_suffix =label_suffix)  # 执行计算mIoU的函数

                temp_miou = np.nanmean(IoUs) * 100

                self.mious.append(temp_miou)
                self.epoches.append(epoch)

                with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                    f.write(str(temp_miou))
                    f.write("\n")

                plt.figure()
                plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Miou')
                plt.title('A Miou Curve')
                plt.legend(loc="upper right")

                plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
                plt.cla()
                plt.close("all")

                print("Get miou done.")
                shutil.rmtree(self.out_path)

            if self.classification and not self.segmentation:
                gt_dir = os.path.join(self.dataset_path, self.dataset, "ImageSets/Classification")
                if not os.path.exists(self.out_path ):
                    os.makedirs(self.out_path )
                labels = np.array(pd.read_csv(os.path.join(gt_dir, 'val_labels.txt'), header =None, sep='\t').iloc[:,1], dtype=np.int16)
                print("Get Overall Accuracy.")
                preds = []
                for image_id in tqdm(self.image_ids):

                    image_path = os.path.join(self.dataset_path, self.dataset, "JPEGImages/" + image_id + ".jpg")
                    image = Image.open(image_path)

                    pred = self.get_pred(image)
                    preds.append(pred)

                print("Calculate Overall Accuracy.")
                _, oa, _,_, kappa = compute_OA_Kappa(labels, np.array(preds),  self.num_classes_cls, None)  # 执行计算mIoU的函数


                self.OA.append(oa)
                self.kappa.append(kappa)
                self.epoches.append(epoch)

                with open(os.path.join(self.log_dir, "epoch_percision.txt"), 'a') as f:
                    f.write(str(oa) + '\t' + str(kappa))
                    f.write("\n")

                plt.figure()
                plt.plot(self.epoches, self.OA, 'red', linewidth = 2, label='train OA')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('OA')
                plt.title('A OA Curve')
                plt.legend(loc="upper right")

                plt.savefig(os.path.join(self.log_dir, "epoch_OA.png"))
                plt.cla()
                plt.close("all")

                print("Get OA done.")
                shutil.rmtree(self.out_path)
            if self.segmentation and self.classification:
                gt_dir = os.path.join(self.dataset_path, self.dataset, "SegmentationClass")
                pred_dir = os.path.join(self.out_path, self.dataset, 'detection-results')

                if not os.path.exists(self.out_path ):
                    os.makedirs(self.out_path )
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                print("Get miou.")
                for image_id in tqdm(self.image_ids):

                    image_path  = os.path.join(self.dataset_path, self.dataset, "JPEGImages/"+image_id+".jpg")
                    image       = Image.open(image_path)
                    if dataset_type == 'RS':
                        tif = None
                    elif dataset_type == 'RSB':
                        tif_path = os.path.join(os.path.join(self.dataset_path, self.dataset, "BuldingFeatures"),
                                                image_id + ".tif")
                        tif = gdal.Open(tif_path)
                        tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                        tif = np.expand_dims(np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                                             0)  # (H,W,C) => (1,C,H,W)
                    elif dataset_type == 'RSP':
                        tif_path = os.path.join(os.path.join(self.dataset_path, self.dataset, "POIFeatures"),
                                                image_id + ".tif")
                        tif = gdal.Open(tif_path)
                        tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize, buf_xsize=2000, buf_ysize=2000)
                        tif = np.expand_dims(np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                                             0)

                    image = self.get_pred(image, tif)
                    save_path = os.path.join(pred_dir, image_id + ".png")
                    if not os.path.exists(save_path):
                        save_dir = os.path.join(pred_dir, image_id.split('/')[0])
                        os.makedirs(save_dir, exist_ok=True)
                    image.save(save_path)

                print("Calculate miou.")

                _, IoUs, _, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes_seg, None)  # 执行计算mIoU的函数

                temp_miou = np.nanmean(IoUs) * 100

                self.mious.append(temp_miou)
                self.epoches.append(epoch)

                with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                    f.write(str(temp_miou))
                    f.write("\n")

                plt.figure()
                plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Miou')
                plt.title('A Miou Curve')
                plt.legend(loc="upper right")

                plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
                plt.cla()
                plt.close("all")

                print("Get miou done.")
                shutil.rmtree(self.out_path)