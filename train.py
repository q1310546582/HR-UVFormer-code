import argparse
import os
import datetime
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.SlWinSegformer import SlWinSegformer
from nets.MixerSegformer import MixerSegformer
from nets.SwinSegformer import SwinSegformer
from nets.HR_UVFormer import HR_UVFormer
from utils.get_metrics import get_metrics
from nets.segformer import SegFormer
from nets.segformer_training import (get_lr_scheduler, set_optimizer_lr,
                                 weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import SegmentationDataset, seg_dataset_collate, ClassificationDataset, \
    class_dataset_collate, ClsAndSegDataset, cls_seg_dataset_collate, collate_fn
from utils.utils import show_config, model_init_resume
from utils.utils_fit import fit_one_epoch


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    #---------------------------------#
    #   Dataset params
    #---------------------------------#
    parser.add_argument("--dataset", choices=["DenseHouse_v1", "DenseHouse_v2"], default="DenseHouse_v1",
                        help="v1 for Segformer, v2 for other models")
    parser.add_argument("--dataset_dir", default="VOCdevkit",
                        help="The path of the dataset. The default is ./VOCdevkit")
    parser.add_argument("--dataset_type", default="RS", choices=["RS", "RSB", "RSP"],
                        help="Options for multimodal data sets, RSB represents image and building dataset, RSPrepresents image and POI dataset")
    #---------------------------------#
    #   Model params
    #---------------------------------#
    parser.add_argument("--model_type", choices=["Segformer", "MixerSegformer", "SwinSegformer", "SlWinSegformer","HR_UVFormer"], type=str,
                        default="Segformer",
                        help="Network structure to be selected")
    parser.add_argument("--backbone_type", choices=["b0", "b1", "b2",
                                                 "b3", "b4", "b5"],
                        default="b0", type=str,
                        help="Which backbone to use for Segformer.")
    parser.add_argument("--output_type", choices=[ "segmentation", "cls_to_seg"], type=str,
                        default="segmentation",
                        help="Set the output of the model, Segforemer is only set to cls _to_seg, other models are only set to Segmentation")
    parser.add_argument("--num_classes_cls", type=int, default=3,
                        help="Num_classes for classification. This parameter is valid only for Segformer and is used to set the number of output categories of the Classifier")
    parser.add_argument("--num_classes_seg", type=int, default=2,
                        help="Num_classes for segmentation.Segformer set to 2, other models set to 3")
    parser.add_argument("--pretrained_filename", type=str, default="",
                        help="Where to search for pretrained models weights filename.")
    parser.add_argument("--pretrained_backbone", action='store_true',
                        help="Whether the pretrained_filenam file is used to match only the backbone part of the model")

    parser.add_argument("--output_dir", default="outputs", type=str,
                        help="The output directory where weights filename will be written.")

    parser.add_argument("--input_shape", default=200, type=int,
                        help="Resolution size for model input, The size of the input picture will be scaled to this value"
                             "Specify 2000 when using the Hierarchical Network, Specify 200 when using the Segformer")
    parser.add_argument("--patch_size", default=200, type=int,
                        help="When using a hierarchical network, used to specify the size of the division into Patches")

    #---------------------------------#
    #   Training strategy
    #---------------------------------#
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Optimizer.")
    parser.add_argument("--optimizer_type", choices= ['adam', 'adamw', 'sgd'], default='adamw', type=str,
                        help="The optimizer types.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--focal_loss", action='store_true',
                        help="Whether to use focal_loss.")
    parser.add_argument("--focal_loss_weights", default=[1,8,4],
                        help="List of each class weight.")

    parser.add_argument("--dice_loss", action='store_true',
                        help="Whether to (use dice_loss + cross_entropy_loss) as the loss function for segmentation.")

    parser.add_argument("--init_epoch", default=0, type=int,
                        help="The number of training epochs to start.")
    parser.add_argument("--freeze_epoch", default=10, type=int,
                        help="Freeze training of Epoches, During Freeze, Backbone's parameters are in a frozen state")
    parser.add_argument("--num_epoches", default=50, type=int,
                        help="Total Epoches of training ")


    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="When unfreeze statge: batch size for training.")
    parser.add_argument("--train_freeze_batch_size", default=8, type=int,
                        help="When freeze statge: batch size for training.")

    parser.add_argument("--eval_every_epoches", default=50, type=int,
                        help="Run prediction on validation set every so many epoches."
                             "Will always run one evaluation at the end of epoches training.")
    parser.add_argument("--save_every_epoches", default=1, type=int,
                        help="Save the model every epoches.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--fp16', action='store_true',
                        help="For fp16: auto Apex AMP optimization")

    #---------------------------------#
    #   Evaluation params
    #---------------------------------#

    parser.add_argument('--only_evaluate', action='store_true',
                        help="Calculate the accuracy of the test set only")

    parser.add_argument('--name_classes_file', type=str, default=None,
                        help="Name of each category")

    parser.add_argument('--evaluate_type', type=str, default='segmentation',
                        choices=['classification', 'segmentation', 'cls_to_seg'
                                 ''],
                        help="classification: Calculate the coarse scale extraction accuracy of Segformer"
                             "segmentation: Calculate the pixel-level extraction accuracy of Segformer, or calculate the coarse-scale extraction accuracy of other models"
                             "cls_to_seg: Calculate the fine-grained extraction accuracy of Segformer or another hierarchical extraction model")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = main()
    Project_Name   = args.name + '_' + args.backbone_type
    #---------------------------------#
    # Cuda Whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda            = True if args.local_rank != -1 else False
    #---------------------------------------------------------------------#
    # distributed is used to specify whether to use single multi-card distributed operation
    # CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # The default under Windows is to use DP mode to call all graphics cards, DDP is not supported.
    # DP mode:
    # Set distributed = False
    # Type CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
    # DDP mode:
    # Set distributed = True
    # Type CUDA_VISIBLE_DEVICES=0,1 in terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn Whether to use sync_bn, DDP mode multi-card available
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    # fp16 whether to use mixed precision training
    # Can reduce video memory by about half, requires pytorch 1.7.1 or higher
    #---------------------------------------------------------------------#
    fp16            = args.fp16

    num_classes_cls     = args.num_classes_cls
    num_classes_seg     = args.num_classes_seg
    #-------------------------------------------------------------------#
    # Backbone networks used:
    # b0, b1, b2, b3, b4, b5
    #-------------------------------------------------------------------#
    phi             = args.backbone_type

    #----------------------------------------------------------------------------------------------------------------------------#
    # Pre-training weights file
    # Do not load the entire model weights when model_path = ' '
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path = args.pretrained_filename
    #------------------------------#
    # Enter the size of the image
    #------------------------------#
    input_shape     = [args.input_shape, args.input_shape] # Default : 200
    
    #----------------------------------------------------------------------------------------------------------------------------#
    # The training is divided into two phases, the freezing phase and the unfreezing phase. The freeze phase is set to meet the training needs of students with insufficient machine performance.
    # Freeze training requires a small amount of video memory, and in the case of very poor graphics cards, Freeze_Epoch can be set equal to NUM_Epoch, when only freeze training is performed.
    #------------------------------------------------------------------#
    # Backbone does not update parameters during the freeze phase
    # Init_Epoch (used during breakpoint training)
    # Freeze_Epoch Epoches for model freeze training
    # Freeze_batch_size The model freezes the training's batch_size
    #------------------------------------------------------------------#
    Init_Epoch          = args.init_epoch
    Freeze_Epoch        = args.freeze_epoch
    Freeze_batch_size   = args.train_freeze_batch_size
    #------------------------------------------------------------------#
    # All parameters of the network will be updated during the thawing phase
    # Num_Epoch Total epochs trained by the model
    # Unfreeze_batch_size model's batch_size after unfreezing
    #------------------------------------------------------------------#
    Num_Epoch      = args.num_epoches
    Unfreeze_batch_size = args.train_batch_size

    Init_lr             = args.learning_rate
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  adam、adamw、sgd
    # momentum    parameter used internally by the momentum optimizer
    # weight_decay  weight decay to prevent overfitting
    # adam will cause weight_decay error, it is recommended to set to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = args.optimizer_type
    momentum            = 0.9
    weight_decay        = args.weight_decay
    #------------------------------------------------------------------#
    #   lr_decay_type   'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     How many epochs to save the weights once
    #------------------------------------------------------------------#
    save_period         = args.save_every_epoches
    #------------------------------------------------------------------#
    #    Dataset path and dataset name
    #------------------------------------------------------------------#
    VOCdevkit_path     = args.dataset_dir
    dataset            = args.dataset
    # ------------------------------------------------------------------#
    #   save_dir
    # The folder where permissions and log files are saved
    #------------------------------------------------------------------#
    save_dir            = args.output_dir
    #------------------------------------------------------------------#
    #   eval_flag       Whether or not to evaluate at training time for the validation set
    #   eval_period     represents how many epochs to evaluate once
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = args.eval_every_epoches

    #------------------------------------------------------------------#
    # Suggested options: dice_loss is a loss function similar to IOU
    # Set to True when there are few types (a few)
    # When there are many types (more than a dozen), if batch_size is large (10 or more), then set to True
    # When there are many species (a dozen or so), if batch_size is small (under 10), then set to False
    #------------------------------------------------------------------#
    dice_loss       = args.dice_loss

    # Whether to assign different loss weights to different classes, default is balanced.
    # If set, note that it is set to numpy form, with the same length as num_classes.
    # For example:
    # num_classes = 3
    # cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    if args.output_type == 'classification':
        cls_weights     = np.ones([args.num_classes_cls], np.float32)
    else:
        cls_weights = np.ones([args.num_classes_seg], np.float32)
    #------------------------------------------------------------------#
    # Whether to use focal loss to prevent positive and negative sample imbalance
    #------------------------------------------------------------------#
    focal_loss      = args.focal_loss
    if focal_loss:
        cls_weights = np.array(args.focal_loss_weights, np.float32)

    #------------------------------------------------------------------#
    # num_workers is used to set whether to use multi-threaded read data, 1 means close multi-threaded
    #------------------------------------------------------------------#
    num_workers     = 4

    #------------------------------------------------------#
    # Set up the graphics card used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = args.local_rank

    #------------------------------------------------------#
    #------------------------------------------------------#
    if args.output_type == 'classification':
        classification = True
        segmentation   = False
    elif args.output_type == 'segmentation':
        classification = False
        segmentation   = True
    elif args.output_type == 'cls_to_seg':
        classification = True
        segmentation   = True

    patch_size = args.patch_size
    if args.model_type == 'Segformer':
        model   = SegFormer(num_classes_seg=num_classes_seg, num_classes_cls=num_classes_cls, phi=phi, classification=classification, segmentation=segmentation, input_shape=patch_size)
    elif args.model_type == 'MixerSegformer':
        backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=phi, classification=True, segmentation=True if args.only_evaluate else False, input_shape=patch_size)
        model    = MixerSegformer(backbone, input_shape=input_shape[0], patch_size=patch_size, num_classes=num_classes_cls, num_blocks=4, token_hidden_dim=256,
                        channel_hidden_dim=2048)
    elif args.model_type == 'SwinSegformer':
        backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=phi, classification=True, segmentation=True if args.only_evaluate else False, input_shape=patch_size)
        model    = SwinSegformer(backbone, input_shape=input_shape[0], patch_size=patch_size, num_classes=num_classes_cls)
    elif args.model_type == 'SlWinSegformer':
        backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=phi, classification=True, segmentation=True if args.only_evaluate else False, input_shape=patch_size)
        model = SlWinSegformer(backbone, input_shape=input_shape[0], patch_size=patch_size, window_size = 5, num_classes=num_classes_cls)
    elif args.model_type == 'HR_UVFormer':
        backbone = SegFormer(num_classes_seg=2, num_classes_cls=3, phi=phi, classification=True,
                                   segmentation=True if args.only_evaluate else False, input_shape=patch_size)
        backbone_build = SegFormer(num_classes_seg=2, num_classes_cls=3, phi='b0', classification=True,
                                   segmentation=False, input_shape=patch_size, input_channels = 3)
        # Initialize branch network with b0 pre-training weights file
        model_init_resume(backbone_build, 'model_data/segformer_b0_weights_voc.pth', local_rank, False, device)
        model = HR_UVFormer(backbone, input_shape=input_shape[0], patch_size=patch_size, window_size = 5, num_classes=num_classes_cls, building_bone = backbone_build)

    #---------------------------#
    # Model initialization with loading the model as a whole or backbone pre-trained weight file
    #---------------------------#
    model_init_resume(model, model_path, local_rank, args.pretrained_backbone, device)
    #----------------------#
    # Record Loss
    #----------------------#
    if local_rank <= 0 and not args.only_evaluate:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, Project_Name + "_loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None


    #----------------------------#
    # Multi-card synchronization Bn
    #----------------------------#
    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_parallel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:

        if distributed:
            #----------------------------#
            # Multiple cards running in parallel
            #----------------------------#
            model_train.cuda(local_rank)
            model_parallel = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_parallel = torch.nn.DataParallel(model_train)
            cudnn.benchmark = True
            model_parallel = model_parallel.cuda()

        model_train = model_parallel
    
    #---------------------------#
    # Read the txt corresponding to the dataset
    #---------------------------#

    with open(os.path.join(VOCdevkit_path, dataset, f"ImageSets/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, dataset, f"ImageSets/val.txt"),"r") as f:
        val_lines = f.readlines()

    num_train   = len(train_lines)
    num_val     = len(val_lines)
    dataset_type = args.dataset_type
    # ----------------------#
    # Evaluation only
    # ----------------------#

    if args.only_evaluate:
        with open(args.name_classes_file, "r") as f:
            name_classes = [line.strip() for line in f.readlines()]

        evaluate_type = args.evaluate_type

        get_metrics(model_train, input_shape, Cuda, num_classes_cls, num_classes_seg, val_lines, VOCdevkit_path, dataset,
                    evaluate_type=evaluate_type, name_classes=name_classes, dataset_type=dataset_type)
        sys.exit()

    #---------------------------#
    # Calculate the number of model parameters
    #---------------------------#
    params = sum([param.numel() for param in model.parameters()])

    if local_rank <= 0:
        show_config(
            num_classes = "classification:%d, segmentation:%d" % (num_classes_cls, num_classes_seg), phi = phi, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, Num_Epoches = Num_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Epoches = Freeze_Epoch, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val,
            params = "%.2fM" % (params / 1e6)
        )

    Freeze_Train = True if Freeze_Epoch > 0 and Init_Epoch < Freeze_Epoch else False
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        # Freeze a certain part of training
        #------------------------------------#
        if Freeze_Train:
            for name,param in model.backbone.named_parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # Determine the current batch_size and adjust the learning rate adaptively
        #-------------------------------------------------------------------#
        nbs             = 16

        lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
        lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Num_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue the training, please expand the dataset.")

        if segmentation and not classification:
            train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes_seg, True, VOCdevkit_path, dataset, dataset_type)
            val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes_seg, False, VOCdevkit_path, dataset, dataset_type)
        elif classification and not segmentation:
            train_dataset   = ClassificationDataset(train_lines, input_shape, num_classes_cls, True, VOCdevkit_path, dataset, dataset_type)
            val_dataset     = ClassificationDataset(val_lines, input_shape, num_classes_cls, False, VOCdevkit_path, dataset, dataset_type)
        elif segmentation and classification:
            train_dataset   = ClsAndSegDataset(train_lines, input_shape, num_classes_cls, num_classes_seg, True, VOCdevkit_path, dataset, dataset_type)
            val_dataset     = ClsAndSegDataset(val_lines, input_shape, num_classes_cls, num_classes_seg,  False, VOCdevkit_path, dataset, dataset_type)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=collate_fn(classification, segmentation)
                                  )
        val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   collate_fn=collate_fn(classification, segmentation)
                                )

        #----------------------#
        # Record eval's map curve
        #----------------------#
        if local_rank <= 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes_cls, num_classes_seg, val_lines, VOCdevkit_path, dataset, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period, classification = classification, segmentation = segmentation)
        else:
            eval_callback   = None

        # Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, Num_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train: # Conditions for the end of the freeze training phase
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                # Determine the current batch_size and adjust the learning rate adaptively
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Num_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size,
                                             num_workers=num_workers, pin_memory=True,
                                             drop_last=True,
                                             collate_fn=collate_fn(classification, segmentation),
                                             sampler=train_sampler)
                val_loader = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size,
                                           num_workers=num_workers, pin_memory=True,
                                           drop_last=True,
                                           collate_fn=collate_fn(classification, segmentation),
                                           sampler=val_sampler)


                UnFreeze_flag   = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_loader, val_loader, Num_Epoch, Cuda, \
                dice_loss, focal_loss, cls_weights, num_classes_cls, num_classes_seg, fp16, scaler, save_period, save_dir, local_rank, classification, segmentation,
                Project_Name, dataset_type)

            if distributed:
                dist.barrier()

        if local_rank <= 0:
            loss_history.writer.close()


