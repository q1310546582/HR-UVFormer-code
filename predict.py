#----------------------------------------------------#
#   将单张图片预测并可视化
#----------------------------------------------------#
import argparse
import os
import torch

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal

from segformer import Model
from utils.utils import preprocess_input_bud, preprocess_input_poi
from utils.utils_metrics import compute_mIoU, show_results

"""
--input_dir VOCdevkit/DenseHouse/JPEGImages
--gt_dir VOCdevkit/DenseHouse/SegmentationClass
--backbone_type b0
--decoder_type classification
--num_classes 3
--pretrained_filename outputs/Segformer_UV_Classification_DenseHouse_b0/ep050-loss0.067-val_loss0.016.pth
--local_rank 0
--classification_replace_segmentation
--name_classes_file VOCdevkit/DenseHouse/ImageSets/Segmentation/name_classes.txt
"""
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset_dir", required=True, type=str,
                        help="directory of dataset ")
    parser.add_argument("--save_dir",  type=str, default='',
                        help="Path of real sample")
    parser.add_argument("--model_type", choices=["Segformer", "MixerSegformer", "SwinSegformer", "SlWinSegformer", "HR_UVFormer"], type=str,
                        default="Segformer",
                        help="Results to be output by the model.")
    parser.add_argument("--dataset_type", default="RS", choices=["RS", "RSB", "RSP"],
                        help="Options for multimodal data sets, RSB represents image and building data set")

    parser.add_argument("--input_shape", default=200, type=int,
                        help="Resolution size for model input, The size of the input picture will be scaled to this value"
                             "Specify 2000 when using the layered network, Specify 200 when using the Segformer")
    parser.add_argument("--patch_size", default=200, type=int,
                        help="When using hierarchical network, the original size of each patch image should be input_ Integer multiple of shape")

    parser.add_argument("--backbone_type", choices=["b0", "b1", "b2",
                                                 "b3", "b4", "b5"],
                        default="b0", type=str,
                        help="Which backbone to create model.")
    parser.add_argument("--output_type", choices=["classification", "segmentation", "cls_to_seg"], type=str,
                        default="segmentation",
                        help="Results to be output by the model.")

    parser.add_argument("--num_classes_cls", type=int, default=3,
                        help="Num_classes for classification.")
    parser.add_argument("--num_classes_seg", type=int, default=2,
                        help="Num_classes for segmentation.")

    parser.add_argument("--pretrained_filename", type=str, default="model_data/segformer_b0_backbone_weights.pth",
                        help="Where to search for pretrained Segformer models filename.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--mode', type=str, default='predict',
                        help="Specify whether the model inference sample or image")

    parser.add_argument('--evaluate_type', type=str, default='segmentation',
                        choices=[ 'segmentation', 'cls_replace_seg', 'cls_to_seg'
                                 ''],
                        help="The accuracy evaluation method adopted by the current model. "
                             "If the current model can only be used for classification, the options are limited to [ cls_replace_seg]"
                             "If the current model can only be used for division, the options are limited to[segmentation]"
                             "If the current model can be classified to divided, the options are limited to[ cls_to_seg]")


    parser.add_argument('--name_classes_file', type=str, default=None,
                        help="Name of each category")

    args = parser.parse_args()

    return args

def arr2img(save_path, arr, width, height, band, transform=None, projection=None):
    # Save as TIF format
    driver = gdal.GetDriverByName("GTiff")
    datasetnew = driver.Create(save_path, width, height, band, gdal.GDT_Float32, ['COMPRESS=PACKBITS'])
    #     datasetnew.WriteRaster(0,0,width,height,arr.tobytes(),width,height,band_list=[1,2,3,4])
    for i in range(band):
        datasetnew.GetRasterBand(i + 1).WriteArray(arr[i])
    if transform != None:
        datasetnew.SetGeoTransform(transform)
    if projection != None:
        datasetnew.SetProjection(projection)
    datasetnew.FlushCache()  # Write to disk

if __name__ == "__main__":
    args = main()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi             = args.backbone_type
    Cuda            = True if args.local_rank != -1 else False
    model_path      = args.pretrained_filename
    model_type      = args.model_type
    input_shape     = args.input_shape
    patch_size      = args.patch_size
    num_classes_cls = args.num_classes_cls
    num_classes_seg = args.num_classes_seg

    if args.output_type == 'classification':
        classification = True # model output for classification
        segmentation   = False # model output for segmentation
    elif args.output_type == 'segmentation':
        classification = False
        segmentation   = True
    elif args.output_type == 'cls_to_seg':
        classification = True
        segmentation   = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = args.local_rank
    evaluate_type = args.evaluate_type

    mix_type = 0 # Hybrid approach to output and prediction

    model = Model(model_path, model_type, num_classes_seg, num_classes_cls, phi, input_shape, patch_size, mix_type, Cuda,
                      classification, segmentation)

    miou_out_path   = ".temp_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)

    print("Get predict result.")

    mode = args.mode

    count           = False
    with open(args.name_classes_file, "r") as f:
        name_classes = [line.strip() for line in f.readlines()]

    dataset_path = args.dataset_dir
    dir_save_path   = args.save_dir


    if mode == "predict":

        print('current input images directory:', dataset_path)
        while True:
            img_num = input('Input image number:')
            if input_shape!= 2000: # for DenseHouse_V2
                name_list = [name.split('.')[0] for name in os.listdir(os.path.join(dataset_path, 'JPEGImages', img_num))]
                sample_size = 2000
                patch_size  = args.patch_size
                steps = sample_size // patch_size
                patch_rows = [name_list[i*steps:(i+1)*steps] for i in range(steps)] # patch rows of list
                rows_predict  = []
                rows_label    = []
                for row in patch_rows:
                    line_predict = []
                    line_label   = []
                    for patch_input in row:
                        image = Image.open(os.path.join(dataset_path, 'JPEGImages', img_num, patch_input + '.jpg'))
                        label = Image.open(os.path.join(dataset_path, 'SegmentationClass', img_num, patch_input + '.png'))
                        # Fusion result of prediction and input image
                        r_image,predict = model.detect_image(image, count=count, name_classes=name_classes,
                                                        evaluate_type = evaluate_type, input_shape = input_shape)
                        predict_array = np.array(r_image)
                        line_predict.append(predict_array)
                        line_label.append(np.array(label))
                        save_dir = os.path.join(pred_dir, img_num)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        predict.save(os.path.join(save_dir, patch_input + '.png')) # 保存预测结果到临时文件夹
                    rows_predict.append(np.concatenate(line_predict, axis=1))
                    rows_label.append(np.concatenate(line_label, axis=1))
                sample_image = np.concatenate(rows_predict, axis=0)
                sample_label = np.concatenate(rows_label, axis=0)
            else: # for DenseHouse_V1
                name_list      = [str(img_num)]
                img_filename   = img_num + '_image.jpg'
                label_filename = img_num + '_image_pixel.png'
                image = Image.open(os.path.join(dataset_path, 'JPEGImages', img_filename))
                sample_label = np.array(Image.open(os.path.join(dataset_path, 'SegmentationClass', label_filename)))

                if args.dataset_type == 'RS':
                    tif = None
                elif args.dataset_type == 'RSB':
                    tif_path = os.path.join(os.path.join(dataset_path, "BuldingFeatures"),
                                            img_num + "_image.tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize,buf_xsize=2000,buf_ysize=2000)

                    tif = np.expand_dims(
                        np.transpose(preprocess_input_bud(np.array(tif, np.float32)), [2, 0, 1]),
                        0)  # (H,W,C) => (1,C,H,W)

                elif args.dataset_type == 'RSP':
                    tif_path = os.path.join(os.path.join(dataset_path, "POIFeatures"),
                                            img_num + "_image.tif")
                    tif = gdal.Open(tif_path)
                    tif = tif.ReadAsArray(0, 0, tif.RasterXSize, tif.RasterYSize,buf_xsize=2000,buf_ysize=2000)

                    tif = np.expand_dims(
                        np.transpose(preprocess_input_poi(np.array(tif, np.float32)), [2, 0, 1]),
                        0)

                r_image,predict = model.detect_image(image, tif=tif, count=count, name_classes=name_classes,
                                             evaluate_type=evaluate_type, input_shape = input_shape)
                sample_image = np.array(r_image)

                save_dir = os.path.join(pred_dir, 'predict')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                predict.save(os.path.join(save_dir, img_num + '.png'))

            sample_label[sample_label>0] = 255
            # Label drawing edges
            sobelx = cv2.Sobel(sample_label, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(sample_label, cv2.CV_64F, 0, 1)
            sobelx = cv2.convertScaleAbs(sobelx)
            sobely = cv2.convertScaleAbs(sobely)
            sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
            boundary = cv2.dilate(np.uint8(sobelxy>0)*255,  np.ones((4, 4), np.uint8)) # Edge swelling treatment
            sample_image[boundary>0] = (227, 252, 8)

            # Get miou
            gt_dir = os.path.join(dataset_path, 'SegmentationClass', img_num) if input_shape != 2000 else  os.path.join(dataset_path, 'SegmentationClass')
            print("Get miou.")
            pred_dir = os.path.join(pred_dir, img_num) if input_shape != 2000 else os.path.join(pred_dir, 'predict')
            if input_shape\
                    == 2000:
                label_suffix = '_image_pixel'
            else:
                label_suffix = ''
            hist, IoUs, PA_Recall, Precision, _ = compute_mIoU(gt_dir, pred_dir, name_list, num_classes_seg,
                                                            name_classes, label_suffix)  # 执行计算mIoU的函数
            print("Get miou done.")
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

            cv2.putText(sample_image, f"mIOU:{np.mean(IoUs):.2f}", (850, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
            # cv2.putText(sample_image, f"mIOU:0.83", (850, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)

            sample_image = Image.fromarray(np.uint8(sample_image))
            sample_image.show()
            if dir_save_path != '':
                sample_image.save(os.path.join(dir_save_path, img_num) + '.jpg')

    elif mode == "interpret":
        image_path = input('Input image filename:')
        patch_size = 2000

        labels_sav_dir = os.path.join('.temp_out', 'interpret')
        if dir_save_path != '':
            labels_sav_dir = dir_save_path

        dataset = gdal.Open(image_path)
        transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        rows = dataset.RasterYSize // patch_size
        cols = dataset.RasterXSize // patch_size

        if cols * patch_size < dataset.RasterXSize:
            left_cols = dataset.RasterXSize - cols * patch_size
            padding_cols = patch_size - left_cols
        if rows * patch_size < dataset.RasterYSize:
            left_rows = dataset.RasterYSize - rows * patch_size
            padding_rows = patch_size - left_rows

        image = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize)
        image = np.pad(image, [(0, 0), (0, padding_rows), (0, padding_cols)], 'constant')
        new_label = np.zeros([1, image.shape[1], image.shape[2]], dtype=np.uint8)

        import time

        T1 = time.time()
        for row in range(rows + 1):
            for col in range(cols + 1):
                patch_image = image[:3, row * patch_size: (row + 1) * patch_size,
                              col * patch_size: (col + 1) * patch_size]
                patch_image = Image.fromarray(np.transpose(patch_image,(1,2,0)))
                _, pr = model.detect_image(patch_image, tif=None, count=count, name_classes=name_classes,
                                                      evaluate_type=evaluate_type, input_shape=input_shape)

                new_label[:, row * patch_size: (row + 1) * patch_size,
                col * patch_size: (col + 1) * patch_size] = np.array(pr)

        T2 = time.time()
        print('%sms' % ((T2 - T1) * 1000))

        new_label = new_label[:, :dataset.RasterYSize, :dataset.RasterXSize]
        arr2img(os.path.join(labels_sav_dir, evaluate_type + os.path.basename(image_path)), new_label, dataset.RasterXSize,
                dataset.RasterYSize, 1, transform, projection)