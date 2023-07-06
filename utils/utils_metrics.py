import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5, classification = False, segmentation = False):
    logits_class, logits_seg = inputs
    if classification and not segmentation:
        # (B,C)
        temp_inputs = torch.softmax(logits_class, -1)

        # --------------------------------------------#
        #   Calculate F1-Score
        # --------------------------------------------#
        temp_inputs = torch.gt(temp_inputs, threhold).float()
        tp = torch.sum(target * temp_inputs, axis=0)  # (C)
        fp = torch.sum(temp_inputs, axis=0) - tp # (C)
        fn = torch.sum(target, axis=0) - tp # (C)

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = torch.mean(score)
    elif segmentation and not classification:
        n, c, h, w = logits_seg.size()
        nt, ht, wt, ct = target.size()

        # (B,C,H,W)=>softmax(B,H*W,C)
        temp_inputs = torch.softmax(logits_seg.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
        temp_target = target.view(n, -1, ct)

        #--------------------------------------------#
        #   Calculate the dice factor
        #--------------------------------------------#
        temp_inputs = torch.gt(temp_inputs, threhold).float()
        tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1]) # (B*H*W,C)
        fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
        fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = torch.mean(score)

    elif classification and segmentation:
        # (B,C)
        temp_inputs = torch.softmax(logits_class, -1)

        # --------------------------------------------#
        #   Calculate F1-Score
        # --------------------------------------------#
        temp_inputs = torch.gt(temp_inputs, threhold).float()
        tp = torch.sum(target[0] * temp_inputs, axis=0)  # (C)
        fp = torch.sum(temp_inputs, axis=0) - tp # (C)
        fn = torch.sum(target[0], axis=0) - tp # (C)

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score_cls = torch.mean(score)

        n, c, h, w = logits_seg.size()
        nt, ht, wt, ct = target[1].size()
        if h != ht and w != wt:
            logits_seg = F.interpolate(logits_seg, size=(ht, wt), mode="bilinear", align_corners=True)
        # (B,C,H,W)=>softmax(B,H*W,C)
        temp_inputs = torch.softmax(logits_seg.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
        temp_target = target[1].view(n, -1, ct)

        #--------------------------------------------#
        #   Calculate the dice factor
        #--------------------------------------------#
        temp_inputs = torch.gt(temp_inputs, threhold).float()
        tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1]) # (B*H*W,C)
        fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
        fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score_seg = torch.mean(score)

        score = 0.5*score_cls + 0.5*score_seg

    return score

def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    # a is the label transformed into a one-dimensional array, shape(H×W,); b is the predicted result transformed into a one-dimensional array, shape(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)

    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)  # IOU

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)  # recall of every class

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) # precision of every class

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def Kappa(hist):
    oa = per_Accuracy(hist)
    oe = np.sum(np.sum(hist,0)*np.sum(hist,1)) / np.sum(hist)**2
    return (oa - oe) / (1-oe)

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, label_suffix = ''):
    print('Num classes', num_classes)  
    #-----------------------------------------#
    # Create a matrix that is all zeros, a confusion matrix
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    
    #------------------------------------------------#
    # Get a list of paths to the validation set labels for direct reading
    # Get a list of paths to the segmentation results of the validation set images for direct reading
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + label_suffix + ".png") for x in png_name_list]
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]

    #------------------------------------------------#
    # Read each (picture-tag) pair
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        #------------------------------------------------#
        # Read an image segmentation result and transform it into a numpy array
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))   # (H,W)
        #------------------------------------------------#
        # Read a corresponding label and transform it into a numpy array
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))   # (H,W)

        # If the image segmentation result is not the same size as the label, this image is not counted

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        # Calculate the confusion hist matrix for an image and accumulate
        #------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # Output the average mIoU value of all categories in the currently calculated images for every 10 images
        if name_classes is not None and ind > 0 and ind % 100 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)), # mIOU for every class  in 10 images
                    100 * np.nanmean(per_class_PA_Recall(hist)), # mRecall for every class in 10 images
                    100 * per_Accuracy(hist) # OA with 10 images
                )
            )
    #------------------------------------------------#
    # Calculate the class-by-class mIoU values for all validation set images
    #------------------------------------------------#
    IoUs        = per_class_iu(hist) # # mIOU for every class
    PA_Recall   = per_class_PA_Recall(hist) # mRecall for every class
    Precision   = per_class_Precision(hist) # mPrecision for every class
    #------------------------------------------------#
    # Output the mIoU values category by category
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    # Find the average mIoU value for all categories on all validation set images, ignoring NaN values in the calculation
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Overall Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))


    if np.sum(hist, axis=1).flatten().all() == False: print('WARING: Some sample labels are missing, please check Whether the num_classes parameter is correct')
    return np.array(hist, np.int), IoUs, PA_Recall, Precision, None

def compute_OA_Kappa(labels, preds, num_classes, name_classes=None):
    print('Num classes', num_classes)
    # -----------------------------------------#
    # Create a matrix that is all zeros, a confusion matrix
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    # Create confusion matrix
    # ------------------------------------------------#
    hist += fast_hist(labels.flatten(), preds.flatten(), num_classes)
    #------------------------------------------------#
    # Calculate metrics for all validation set images
    #------------------------------------------------#
    PA_Recall   = per_class_PA_Recall(hist) # mRecall for every class with all images
    Precision   = per_class_Precision(hist) # mPrecision for every class with all images
    OA          = per_Accuracy(hist) # OA with all images
    kappa       = Kappa(hist)

    #------------------------------------------------#
    # Output the following indicators by category
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] +
                  '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+
                  '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    # Find all OA and Kappa coefficients on all validation set images
    #-----------------------------------------------------------------#
    print('===> Overall Accuracy: ' + str(round(OA * 100, 2)) + '; Kappa:' + str(round(kappa * 100, 2)))
    return np.array(hist, np.int), OA, PA_Recall, Precision, kappa



def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            