import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.autopet import (AutoPETDataset, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case

from semisam_plus import semisam_branch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/cyf/codes/SSL4MIS/data/autopet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='autopet/SemiSAM_MT', help='experiment_name')
parser.add_argument('--prompt', type=str,
                    default='unc')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=2,
                    help='labeled data')
parser.add_argument('--modality', type=str, default='pet', 
                    choices=['both', 'ct', 'pet'], help='input modality')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 11
    
    
    if args.modality == 'ct':
        in_chanes = 1
    elif args.modality == 'pet':
        in_chanes = 1
    elif args.modality == 'both':
        in_chanes = 2
    
    

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model, in_chns=in_chanes, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # db_train = BraTS2019(base_dir=train_data_path,
    #                      split='train',
    #                      num=None,
    #                      transform=transforms.Compose([
    #                          RandomRotFlip(),
    #                          RandomCrop(args.patch_size),
    #                          ToTensor(),
    #                      ]))
    
    
    db_train = AutoPETDataset(base_dir=train_data_path,
                         split='train',
                         labeled_num=args.labeled_num, # 👈 新增这一行
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]),
                         modality=args.modality)
    
    
    # db_train = AutoPETDataset(base_dir=train_data_path,
    #                      split='train',
    #                      num=args.labeled_num,
    #                      transform=transforms.Compose([
    #                          RandomRotFlip(),
    #                          RandomCrop(args.patch_size),
    #                          ToTensor(),
    #                      ]),
    #                      modality=args.modality
    #                      )   

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    labeled_idxs = list(range(0, args.labeled_num))
    # unlabeled_idxs = list(range(args.labeled_num, 40))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(11)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print(volume_batch.shape)
            # print(label_batch.shape)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - ema_output_soft)**2)


            sam_con_loss_total = 0.0
            
            
            if args.modality == 'both':
                # SAM-Med3D 只支持单通道，选择 PET 通道（索引1）
                volume_batch = volume_batch[:, 0:1, :, :, :]
            else:
                volume_batch = volume_batch
            
            
            for c in range(1, num_classes):
                current_organ_soft = outputs_soft[:, c:c+1, :, :, :]
                
                
                
                
                
                samseg_mask, uncsam = semisam_branch(
                    volume_batch, 
                    current_organ_soft, 
                    generalist='SAM-Med3D-turbo',
                    prompt=args.prompt
                )
            
                if args.prompt == 'unc':
                    # 计算 MSE 距离
                    dist = (current_organ_soft[args.labeled_bs:] - samseg_mask[args.labeled_bs:]) ** 2
                    
                    # 加权 (如果有不确定性图)
                    # 注意：uncsam 可能也需要对应调整，如果 semisam_branch 内部没处理好，这里简单起见先直接算 MSE
                    # 这里假设 uncsam 是对应当前类别的
                    loss_c = torch.mean(dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(uncsam)
                else:
                    loss_c = torch.mean((current_organ_soft[args.labeled_bs:] - samseg_mask[args.labeled_bs:]) ** 2)
                
                # 累加 Loss
                sam_con_loss_total += loss_c
            
            # 4. 计算最终的加权 Loss
            # 因为累加了 12 次，可能数值会大，可以除以 (num_classes - 1) 做平均，也可以不除
            sam_con_loss_total = sam_con_loss_total / (num_classes - 1)

            consistency_weight_sam = get_current_consistency_weight((args.max_iterations - iter_num )//150)
            
            # 最终 Loss
            sam_con_loss = 0.1 * consistency_weight_sam * sam_con_loss_total
            
            
            # samseg_mask, uncsam = semisam_branch(volume_batch, outputs_soft[:,0:1,:,:,:], generalist='SAM-Med3D-turbo',prompt=args.prompt)
            # samseg_soft = torch.cat((1 - samseg_mask, samseg_mask), dim=1)
            
            # unet_background = outputs_soft[:,0:1,:,:,:]
            # unet_foreground = torch.sum(outputs_soft[:,1:,:,:,:], dim=1, keepdim=True)
            # outputs_soft_binary = torch.cat((unet_background, unet_foreground), dim=1)
            
            
            
            # sam_dice = (label_batch[:args.labeled_bs] - samseg_soft[:args.labeled_bs] ) **2


            # if args.prompt == 'unc':
            #     # 使用 binary 版的 outputs 计算距离
            #     sam_consistency_dist = (outputs_soft_binary[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2
            #     sam_consistency = torch.mean(
            #         sam_consistency_dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(uncsam)
            # else:
            #     sam_consistency = torch.mean(
            #         (outputs_soft_binary[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2)



            # if args.prompt == 'unc':
            #     sam_consistency_dist = (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2
            #     sam_consistency = torch.mean(
            #         sam_consistency_dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(uncsam)
            # else:
            #     sam_consistency = torch.mean(
            #         (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2)


            # consistency_weight_sam = get_current_consistency_weight((args.max_iterations - iter_num )//150)
            # sam_consistency = torch.sum(sam_consistency)/(torch.sum(sam_consistency)+1e-16) 
            # sam_con_loss = 0.1 * consistency_weight_sam * sam_consistency


            loss = supervised_loss + consistency_weight * consistency_loss + sam_con_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 200 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 2000 == 0:
                model.eval()
                val_list_path = os.path.join(f'lists_{args.labeled_num}', 'val.txt')
                avg_metric = test_all_case(
                    model, args.root_path, test_list=val_list_path, num_classes=11, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64,modality=args.modality)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()
                torch.cuda.empty_cache()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
