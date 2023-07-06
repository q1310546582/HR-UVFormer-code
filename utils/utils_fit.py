import os

import torch
from nets.segformer_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_loader, val_loader, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes_cls, num_classes_seg, fp16, scaler, save_period, save_dir, local_rank=0,
                  classification = False, segmentation = False, project_name='', dataset_type = 'RS'):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank <= 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()

    gen = train_loader
    if int(classification) + int(segmentation) == 1:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            imgs, target, tif, labels_one_hot = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda(local_rank)
                    target    = target.cuda(local_rank)
                    tif    = tif.cuda(local_rank) if tif is not None else None
                    labels_one_hot  = labels_one_hot.cuda(local_rank)
                    weights = weights.cuda(local_rank)

            optimizer.zero_grad()
            if not fp16:

                if tif is None:
                    outputs = model_train(imgs)
                else:
                    outputs = model_train(imgs, tif)

                if focal_loss:
                    loss = Focal_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)
                else:
                    loss = CE_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)

                if dice_loss and segmentation:
                    main_dice = Dice_loss(outputs, labels_one_hot)
                    loss      = loss + main_dice

                with torch.no_grad():

                    _f_score = f_score(outputs, labels_one_hot, classification = classification, segmentation = segmentation)

                loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    if tif is None:
                        outputs = model_train(imgs)
                    else:
                        outputs = model_train(imgs, tif)

                    if focal_loss:
                        loss = Focal_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)
                    else:
                        loss = CE_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels_one_hot)
                        loss = loss + main_dice

                    with torch.no_grad():

                        _f_score = f_score(outputs, labels_one_hot, classification = classification, segmentation = segmentation)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss      += loss.item()
            total_f_score   += _f_score.item()

            if local_rank <= 0:
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'f_score'   : total_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
    else:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, target_cls, target_seg, labels_one_hot_cls, labels_one_hot_seg, tif = batch

            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda(local_rank)
                    tif = tif.cuda(local_rank) if tif is not None else None
                    target_cls = target_cls.cuda(local_rank)
                    target_seg = target_seg.cuda(local_rank)
                    labels_one_hot_cls = labels_one_hot_cls.cuda(local_rank)
                    labels_one_hot_seg = labels_one_hot_seg.cuda(local_rank)
                    weights = weights.cuda(local_rank)

            target = [target_cls, target_seg]
            labels_one_hot = [labels_one_hot_cls, labels_one_hot_seg]

            optimizer.zero_grad()
            if not fp16:
                if tif is None:
                    outputs = model_train(imgs)
                else:
                    outputs = model_train(imgs, tif)

                if focal_loss:
                    loss = Focal_Loss(outputs, target, weights, num_classes_seg, classification=classification,
                                      segmentation=segmentation)
                else:
                    loss = CE_Loss(outputs, target, weights, num_classes_seg, classification=classification,
                                   segmentation=segmentation)

                if dice_loss and segmentation:
                    main_dice = Dice_loss(outputs, labels_one_hot)
                    loss = loss + main_dice

                with torch.no_grad():

                    _f_score = f_score(outputs, labels_one_hot, classification=classification,
                                       segmentation=segmentation)

                loss.backward()

                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():

                    if tif is None:
                        outputs = model_train(imgs)
                    else:
                        outputs = model_train(imgs, tif)

                    if focal_loss:
                        loss = Focal_Loss(outputs, target, weights, num_classes_seg,
                                          classification=classification, segmentation=segmentation)
                    else:
                        loss = CE_Loss(outputs, target, weights, num_classes_seg, classification=classification,
                                       segmentation=segmentation)

                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels_one_hot)
                        loss = loss + main_dice

                    with torch.no_grad():

                        _f_score = f_score(outputs, labels_one_hot, classification=classification,
                                           segmentation=segmentation)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            if local_rank <= 0:
                pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                    'f_score': total_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if local_rank <= 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    # ---------------------------------------------#
    # Testing phase
    # ---------------------------------------------#
    model_train.eval()
    gen_val = val_loader

    if int(classification) + int(segmentation) == 1:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            imgs, target, tif, labels_one_hot = batch

            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs    = imgs.cuda(local_rank)
                    target    = target.cuda(local_rank)
                    labels_one_hot  = labels_one_hot.cuda(local_rank)
                    weights = weights.cuda(local_rank)
                    tif = tif.cuda(local_rank) if tif is not None else None

                if tif is None:
                    outputs = model_train(imgs)
                else:
                    outputs = model_train(imgs, tif)

                if focal_loss:
                    loss = Focal_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)
                else:
                    loss = CE_Loss(outputs, target, weights, num_classes_seg, classification = classification, segmentation = segmentation)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels_one_hot)
                    loss  = loss + main_dice

                _f_score    = f_score(outputs, labels_one_hot, classification = classification, segmentation = segmentation)

                val_loss    += loss.item()
                val_f_score += _f_score.item()

            if local_rank <= 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    else:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, target_cls, target_seg, labels_one_hot_cls, labels_one_hot_seg, tif = batch
            with torch.no_grad():
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda(local_rank)
                    target_cls = target_cls.cuda(local_rank)
                    target_seg = target_seg.cuda(local_rank)
                    labels_one_hot_cls = labels_one_hot_cls.cuda(local_rank)
                    labels_one_hot_seg = labels_one_hot_seg.cuda(local_rank)
                    tif = tif.cuda(local_rank) if tif is not None else None
                    weights = weights.cuda(local_rank)

                target = [target_cls, target_seg]
                labels_one_hot = [labels_one_hot_cls, labels_one_hot_seg]
                # ----------------------#
                # Forward propagation
                # ----------------------#
                if tif is None:
                    outputs = model_train(imgs)
                else:
                    outputs = model_train(imgs, tif)
                # ----------------------#
                # Loss calculation
                # ----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, target, weights, num_classes_seg, classification=classification,
                                      segmentation=segmentation)
                else:
                    loss = CE_Loss(outputs, target, weights, num_classes_seg, classification=classification,
                                   segmentation=segmentation)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels_one_hot)
                    loss = loss + main_dice
                # -------------------------------#
                # Calculate f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels_one_hot, classification=classification, segmentation=segmentation)

                val_loss += loss.item()
                val_f_score += _f_score.item()

            if local_rank <= 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)



    if local_rank <= 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train, dataset_type)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        # Save weights
        #-----------------------------------------------#
        if project_name != '':
            model_output_dir = os.path.join(save_dir, project_name)
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
        if save_period == 1:
             torch.save(model.state_dict(), os.path.join(model_output_dir,  'last_checkpoint.pth'))
             with open(os.path.join(model_output_dir,  'record.txt'), 'w') as f:
                 f.write(f'epoch\t{epoch},f_score\t{val_f_score / (iteration + 1)},val_loss\t{val_loss / (iteration + 1)}')

        elif (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(model_output_dir,  'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(model_output_dir,  "best_epoch_weights_loss_%.2f.pth" % (val_loss / epoch_step_val)))


