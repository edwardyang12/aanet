import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os

from utils import utils
from utils.visualization import disp_error_img, save_images, depth_error_img
from metric import d1_metric, thres_metric, bad, mm_error
from math import inf
from warp_ops import apply_disparity_cu

import imageio
import pickle
import numpy as np

onlyObj = True
class Model(object):
    def __init__(self, args, logger, optimizer, aanet, device, start_iter=0, start_epoch=0,
                 best_epe=None, best_epoch=None):
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.aanet = aanet
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch

        self.best_epe = 999. if best_epe is None else best_epe
        self.best_epoch = -1 if best_epoch is None else best_epoch


        self.train_writer = SummaryWriter(self.args.checkpoint_dir)

    def train(self, train_loader):
        args = self.args
        logger = self.logger

        steps_per_epoch = len(train_loader)
        device = self.device

        self.aanet.train()

        if args.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.aanet.apply(set_bn_eval)

        # Learning rate summary
        base_lr = self.optimizer.param_groups[0]['lr']
        offset_lr = self.optimizer.param_groups[1]['lr']
        self.train_writer.add_scalar('base_lr', base_lr, self.epoch + 1)
        self.train_writer.add_scalar('offset_lr', offset_lr, self.epoch + 1)

        last_print_time = time.time()
        baseline = 0.055
        intrinsic =[[1387.095, 0.0, 960.0], [0.0, 1387.095, 540.0], [0.0, 0.0, 1.0]]


        for i, sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]
            gt_depth = []

            if args.dataset_name=='custom_dataset': # going to be depthL_fromR_down if from  custom_dataset
                gt_disp_1 = (baseline*1000*intrinsic[0][0]/2)/(gt_disp*256.)
                gt_disp_1[gt_disp_1==inf]=0
                gt_depth = gt_disp*256.
                gt_disp = gt_disp_1

            if(args.dataset_name == 'custom_dataset_full'):

                # convert to disparity then apply warp ops
                temp = gt_disp*256.
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    temp[x] = (baseline*1000*intrinsic[0][0]/2)/(temp[x])
                    temp[x][temp[x]==inf] = 0

                gt_disp = apply_disparity_cu(temp.unsqueeze(1),temp.type(torch.int))
                gt_disp = torch.squeeze(gt_disp)

                gt_depth = temp
                # convert to gt_depth
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    gt_depth[x] = (baseline*1000*intrinsic[0][0]/2)/(gt_disp[x])
                    gt_depth[x][gt_depth[x]==inf] = 0
                gt_depth = gt_depth.to(self.device)

            mask = (gt_disp > 0.) & (gt_disp < args.max_disp)

            if args.load_pseudo_gt:
                pseudo_gt_disp = sample['pseudo_disp'].to(device)
                pseudo_mask = (pseudo_gt_disp > 0.) & (pseudo_gt_disp < args.max_disp) & (~mask)  # inverse mask

            if not mask.any():
                continue

            pred_disp_pyramid = self.aanet(left, right)  # list of H/12, H/6, H/3, H/2, H

            if args.highest_loss_only:
                pred_disp_pyramid = [pred_disp_pyramid[-1]]  # only the last highest resolution output

            disp_loss = 0
            pseudo_disp_loss = 0
            pyramid_loss = []
            pseudo_pyramid_loss = []

            # Loss weights
            if len(pred_disp_pyramid) == 5:
                pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
            elif len(pred_disp_pyramid) == 4:
                pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
            elif len(pred_disp_pyramid) == 3:
                pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
            elif len(pred_disp_pyramid) == 1:
                pyramid_weight = [1.0]  # highest loss only
            else:
                raise NotImplementedError

            assert len(pyramid_weight) == len(pred_disp_pyramid)
            for k in range(len(pred_disp_pyramid)):
                pred_disp = pred_disp_pyramid[k]
                weight = pyramid_weight[k]

                if pred_disp.size(-1) != gt_disp.size(-1):
                    pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                    pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                    pred_disp = pred_disp.squeeze(1)  # [B, H, W]

                curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                             reduction='mean')
                disp_loss += weight * curr_loss
                pyramid_loss.append(curr_loss)

                # Pseudo gt loss
                if args.load_pseudo_gt:
                    pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                        reduction='mean')
                    pseudo_disp_loss += weight * pseudo_curr_loss

                    pseudo_pyramid_loss.append(pseudo_curr_loss)

            total_loss = disp_loss + pseudo_disp_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.num_iter += 1

            if self.num_iter % args.print_freq == 0:
                this_cycle = time.time() - last_print_time
                last_print_time += this_cycle

                logger.info('Epoch: [%3d/%3d] [%5d/%5d] time: %4.2fs disp_loss: %.3f' %
                            (self.epoch + 1, args.max_epoch, i + 1, steps_per_epoch, this_cycle,
                             disp_loss.item()))

            if self.num_iter % args.summary_freq == 0:
                img_summary = {}
                img_summary['left'] = left
                img_summary['right'] = right
                img_summary['gt_disp'] = gt_disp

                if args.load_pseudo_gt:
                    img_summary['pseudo_gt_disp'] = pseudo_gt_disp

                # Save pyramid disparity prediction
                for s in range(len(pred_disp_pyramid)):
                    # Scale from low to high, reverse
                    save_name = 'pred_disp' + str(len(pred_disp_pyramid) - s - 1)
                    save_value = pred_disp_pyramid[s]
                    img_summary[save_name] = save_value

                pred_disp = pred_disp_pyramid[-1]

                if pred_disp.size(-1) != gt_disp.size(-1):
                    pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                    pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                    pred_disp = pred_disp.squeeze(1)  # [B, H, W]
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)

                save_images(self.train_writer, 'train', img_summary, self.num_iter)

                epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                self.train_writer.add_scalar('train/epe', epe.item(), self.num_iter)
                self.train_writer.add_scalar('train/disp_loss', disp_loss.item(), self.num_iter)
                self.train_writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

                # Save loss of different scale
                for s in range(len(pyramid_loss)):
                    save_name = 'train/loss' + str(len(pyramid_loss) - s - 1)
                    save_value = pyramid_loss[s]
                    self.train_writer.add_scalar(save_name, save_value, self.num_iter)

                d1 = d1_metric(pred_disp, gt_disp, mask)
                self.train_writer.add_scalar('train/d1', d1.item(), self.num_iter)
                thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
                thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
                thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
                self.train_writer.add_scalar('train/thres1', thres1.item(), self.num_iter)
                self.train_writer.add_scalar('train/thres2', thres2.item(), self.num_iter)
                self.train_writer.add_scalar('train/thres3', thres3.item(), self.num_iter)

        self.epoch += 1

        # Always save the latest model for resuming training
        if args.no_validate:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=-1, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='aanet_latest.pth')

            # Save checkpoint of specific epoch
            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.aanet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=-1, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)

    def validate(self, val_loader):
        args = self.args
        logger = self.logger
        logger.info('=> Start validation...')

        if args.evaluate_only is True:
            if args.pretrained_aanet is not None:
                pretrained_aanet = args.pretrained_aanet
            else:
                model_name = 'aanet_best.pth'
                pretrained_aanet = os.path.join(args.checkpoint_dir, model_name)
                if not os.path.exists(pretrained_aanet):  # KITTI without validation
                    pretrained_aanet = pretrained_aanet.replace(model_name, 'aanet_latest.pth')

            logger.info('=> loading pretrained aanet: %s' % pretrained_aanet)
            utils.load_pretrained_net(self.aanet, pretrained_aanet, no_strict=True)

        self.aanet.train()

        num_samples = len(val_loader)
        logger.info('=> %d samples found in the validation set' % num_samples)

        val_epe = 0
        val_d1 = 0
        # val_thres1 = 0
        # val_thres2 = 0
        # val_thres3 = 0
        val_bad1 = 0
        val_bad2 = 0
        val_abs = 0
        val_mm2 = 0
        val_mm4 = 0
        val_mm8 = 0

        val_count = 0

        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')

        num_imgs = 0
        valid_samples = 0

        baseline = 0.055
        intrinsic =[[1387.095, 0.0, 960.0], [0.0, 1387.095, 540.0], [0.0, 0.0, 1.0]]

        for i, sample in enumerate(val_loader):
            if i % 100 == 0:
                logger.info('=> Validating %d/%d' % (i, num_samples))

            left = sample['left'].to(self.device)  # [B, 3, H, W]
            right = sample['right'].to(self.device)
            gt_disp = sample['disp'].to(self.device)  # [B, H, W]
            gt_depth = []
            pred_disp = []

            if args.dataset_name=='custom_dataset': # going to be depthL_fromR_down if from  custom_dataset
                gt_disp_1 = (baseline*1000*intrinsic[0][0]/2)/(gt_disp*256.)
                gt_disp_1[gt_disp_1==inf]=0
                gt_depth = gt_disp*256.
                gt_disp = gt_disp_1

            if(args.dataset_name == 'custom_dataset_full'):

                # convert to disparity then apply warp ops
                temp = gt_disp*256.
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    temp[x] = (baseline*1000*intrinsic[0][0]/2)/(temp[x])
                    temp[x][temp[x]==inf] = 0

                gt_disp = apply_disparity_cu(temp.unsqueeze(1),temp.type(torch.int))
                gt_disp = torch.squeeze(gt_disp)

                gt_depth = temp
                # convert to gt_depth
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    gt_depth[x] = (baseline*1000*intrinsic[0][0]/2)/(gt_disp[x])
                    gt_depth[x][gt_depth[x]==inf] = 0
                gt_depth = gt_depth.to(self.device)

            if(args.dataset_name == 'custom_dataset_sim' or
                args.dataset_name == 'custom_dataset_real'):
                temp = gt_disp*256.
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    temp[x] = (baseline*1000*intrinsic[0][0]/2)/(temp[x])
                    temp[x][temp[x]==inf] = 0

                gt_disp = apply_disparity_cu(temp.unsqueeze(1),temp.type(torch.int))
                gt_disp = torch.squeeze(gt_disp)

                gt_disp = torch.unsqueeze(gt_disp,0)

                gt_depth = temp
                # convert to gt_depth
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    gt_depth[x] = (baseline*1000*intrinsic[0][0]/2)/(gt_disp[x])
                    gt_depth[x][gt_depth[x]==inf] = 0
                gt_depth = gt_depth.to(self.device)

            mask_disp = (gt_disp > 0.) & (gt_disp < args.max_disp)

            if not mask_disp.any():
                continue

            valid_samples += 1

            num_imgs += gt_disp.size(0)

            with torch.no_grad():
                pred_disp = self.aanet(left, right)[-1]  # [B, H, W]

            if pred_disp.size(-1) < gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            if((args.dataset_name == 'custom_dataset_sim' or
                args.dataset_name == 'custom_dataset_real') and onlyObj):
                gt_disp[sample['label']>=17] = 0
                mask_disp = (gt_disp > 0.) & (gt_disp < args.max_disp)

            epe = F.l1_loss(gt_disp[mask_disp], pred_disp[mask_disp], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask_disp)

            bad1 = bad(pred_disp, gt_disp, mask_disp)
            bad2 = bad(pred_disp, gt_disp, mask_disp, threshold=2)

            pred_depth = []
            if(args.dataset_name == 'custom_dataset_full' or
                args.dataset_name == 'custom_dataset_sim' or
                args.dataset_name == 'custom_dataset_real'):
                temp = torch.zeros((pred_disp.shape)).to(self.device)
                for x in range(left.shape[0]):
                    baseline = sample['baseline'][x].to(self.device)
                    intrinsic = sample['intrinsic'][x].to(self.device)
                    temp[x] = (baseline*1000*intrinsic[0][0]/2)/(pred_disp[x])
                    temp[x][temp[x]==inf] = 0
                pred_depth = temp
            else:
                pred_depth = (baseline*1000*intrinsic[0][0]/2)/(pred_disp)
                pred_depth[pred_depth==inf]=0

            mask_depth = (gt_depth > 0.) & (gt_depth < 2000)

            if((args.dataset_name == 'custom_dataset_sim' or
                args.dataset_name == 'custom_dataset_real') and onlyObj):
                gt_depth[sample['label']>=17] = 0
                mask_depth = (gt_depth > 0.) & (gt_disp < args.max_disp)

            abs = F.l1_loss(gt_depth[mask_depth], pred_depth[mask_depth], reduction='mean')

            mm2 = mm_error(pred_depth, gt_depth,mask_depth)
            mm4 = mm_error(pred_depth, gt_depth,mask_depth, threshold=4)
            mm8 = mm_error(pred_depth, gt_depth,mask_depth, threshold=8)

            pred_depth[pred_depth>2000]=0

            # thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            # thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            # thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_bad1 += bad1.item()
            val_bad2 += bad2.item()
            val_abs += abs.item()
            val_mm2 += mm2.item()
            val_mm4 += mm4.item()
            val_mm8 += mm8.item()
            # val_thres1 += thres1.item()
            # val_thres2 += thres2.item()
            # val_thres3 += thres3.item()

            # Save 3 images for visualization

            if i in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
                if args.evaluate_only:

                    im = (pred_depth[0]).detach().cpu().numpy().astype(np.uint16)
                    if not os.path.isdir('/cephfs/edward/depths'):
                        os.mkdir('/cephfs/edward/depths')
                    imageio.imwrite('/cephfs/edward/depths/'+str(i)+".png",im)

                    im = (gt_depth[0]).detach().cpu().numpy().astype(np.uint16)
                    imageio.imwrite('/cephfs/edward/depths/'+str(i)+"gt.png",im)

                    imageio.imwrite('/cephfs/edward/depths/' + str(i) + "label.png", sample['label'][x].detach().numpy().astype(np.uint8))

                    info = {'baseline': sample['baseline'][x],'intrinsic' :sample['intrinsic'][x],
                        'object_ids': sample['object_ids'][x], 'extrinsic': sample['extrinsic'][x]}
                    filename = '/cephfs/edward/depths/meta' + str(i) + '.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(info, f)

                img_summary = {}
                img_summary['left'] = left
                img_summary['right'] = right
                img_summary['gt_depth'] = gt_depth
                img_summary['gt_disp'] = gt_disp

                if((args.dataset_name == 'custom_dataset_sim' or
                    args.dataset_name == 'custom_dataset_real') and onlyObj):
                    pred_disp[sample['label']>=17] = 0
                    pred_depth[sample['label']>=17] = 0

                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                img_summary['depth_error'] = depth_error_img(pred_depth, gt_depth)
                img_summary['pred_disp'] = pred_disp
                img_summary['pred_depth'] = pred_depth

                save_images(self.train_writer, 'val' + str(val_count), img_summary, self.epoch)
                val_count += 1

        logger.info('=> Validation done!')

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples
        mean_bad1 = val_bad1/valid_samples
        mean_bad2 = val_bad2/valid_samples
        mean_abs = val_abs/valid_samples
        mean_mm2 = val_mm2/valid_samples
        mean_mm4 = val_mm4/valid_samples
        mean_mm8 = val_mm8/valid_samples
        # mean_thres1 = val_thres1 / valid_samples
        # mean_thres2 = val_thres2 / valid_samples
        # mean_thres3 = val_thres3 / valid_samples

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % self.epoch)
            f.write('epe: %.4f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('bad1: %.4f\t' % mean_bad1)
            f.write('bad2: %.4f\t' % mean_bad2)
            f.write('abs: %.4f\t' % mean_abs)
            f.write('mm2: %.4f\t' % mean_mm2)
            f.write('mm4: %.4f\t' % mean_mm4)
            f.write('mm8: %.4f\t' % mean_mm8)
            # f.write('thres1: %.4f\t' % mean_thres1)
            # f.write('thres2: %.4f\t' % mean_thres2)
            # f.write('thres3: %.4f\n' % mean_thres3)

        logger.info('=> Mean validation epe of epoch %d: %.3f' % (self.epoch, mean_epe))

        self.train_writer.add_scalar('val/epe', mean_epe, self.epoch)
        self.train_writer.add_scalar('val/d1', mean_d1, self.epoch)
        self.train_writer.add_scalar('val/bad1', mean_bad1, self.epoch)
        self.train_writer.add_scalar('val/bad2', mean_bad2, self.epoch)
        self.train_writer.add_scalar('val/abs', mean_abs, self.epoch)
        self.train_writer.add_scalar('val/mm2', mean_mm2, self.epoch)
        self.train_writer.add_scalar('val/mm4', mean_mm4, self.epoch)
        self.train_writer.add_scalar('val/mm8', mean_mm8, self.epoch)
        # self.train_writer.add_scalar('val/thres1', mean_thres1, self.epoch)
        # self.train_writer.add_scalar('val/thres2', mean_thres2, self.epoch)
        # self.train_writer.add_scalar('val/thres3', mean_thres3, self.epoch)

        if not args.evaluate_only:
            if args.val_metric == 'd1':
                if mean_d1 < self.best_epe:
                    # Actually best_epe here is d1
                    self.best_epe = mean_d1
                    self.best_epoch = self.epoch

                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_d1, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='aanet_best.pth')
            elif args.val_metric == 'epe':
                if mean_epe < self.best_epe:
                    self.best_epe = mean_epe
                    self.best_epoch = self.epoch

                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_epe, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='aanet_best.pth')
            else:
                raise NotImplementedError

        if self.epoch == args.max_epoch:
            # Save best validation results
            with open(val_file, 'a') as f:
                f.write('\nbest epoch: %03d \t best %s: %.3f\n\n' % (self.best_epoch,
                                                                     args.val_metric,
                                                                     self.best_epe))

            logger.info('=> best epoch: %03d \t best %s: %.3f\n' % (self.best_epoch,
                                                                    args.val_metric,
                                                                    self.best_epe))

        # Always save the latest model for resuming training
        if not args.evaluate_only:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=mean_epe, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='aanet_latest.pth')

            # Save checkpoint of specific epochs
            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.aanet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=mean_epe, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)
