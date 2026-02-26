

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
from torch.utils.data import DataLoader, SubsetRandomSampler  # 修改：引入 SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.autopet import (AutoPETDataset, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics  # 修改：删除 ramps（不再需要）
from val_3D import test_all_case

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/cyf/codes/SSL4MIS/data/autopet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='fully_supervised_autopet', help='experiment_name')  # 修改：默认实验名
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')

# label settings
parser.add_argument('--labeled_num', type=int, default=5,  # 修改：默认使用全部数据（根据你的数据总量调整）
                    help='labeled data number for training (set to total samples for full supervision)')
parser.add_argument('--modality', type=str, default='both',
                    choices=['both', 'ct', 'pet'], help='input modality')

args = parser.parse_args()


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 11  # 1 + 12

    # 设置输入通道数
    if args.modality == 'pet':
        in_chns = 1
    elif args.modality == 'ct':
        in_chns = 1
    elif args.modality == 'both':
        in_chns = 2

    # 1. 只创建学生模型（删除EMA教师模型）
    model = net_factory_3d(net_type=args.model, in_chns=in_chns, class_num=num_classes)
    model = model.cuda()

    # 2. 数据集准备
    db_train = AutoPETDataset(base_dir=train_data_path,
                              split='train',
                              labeled_num=args.labeled_num,
                              transform=transforms.Compose([
                                  RandomRotFlip(),
                                  RandomCrop(args.patch_size),
                                  ToTensor(),
                              ]),
                              modality=args.modality)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 3. 关键修改：使用 SubsetRandomSampler 替代 TwoStreamBatchSampler
    # 只从 0 到 labeled_num-1 采样，不再混合无标签数据
    labeled_idxs = list(range(0, args.labeled_num))
    trainloader = DataLoader(db_train,
                             batch_size=batch_size,
                             sampler=SubsetRandomSampler(labeled_idxs),  # 简单随机采样，只包含有标签数据
                             num_workers=4,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch = label_batch.long()
            # 4. 前向传播（删除EMA相关）
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # 5. 监督损失计算（删除一致性损失）
            # 注意：现在 volume_batch 全是 labeled 数据，不需要 [:args.labeled_bs] 切片
            loss_ce = ce_loss(outputs, label_batch[:])
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 6. 学习率调整（保持不变）
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            # 7. 日志记录
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # 8. 可视化（保持不变）
            if iter_num % 20 == 0:
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

            # 9. 验证（保持不变，每200 iter验证一次）
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                # 注意：验证列表路径可能需要根据你的实际文件结构调整
                # val_list_path = os.path.join(f'lists_{args.labeled_num}', 'val.txt')
                
                # # 如果文件不存在，尝试默认路径
                # # if not os.path.exists(val_list_path):
                # #     val_list_path = os.path.join('lists', 'val.txt')
                    
                # avg_metric = test_all_case(
                #     model, args.root_path, test_list=val_list_path, 
                #     num_classes=num_classes, patch_size=args.patch_size,
                #     stride_xy=64, stride_z=64, modality=args.modality)
                
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
                    logging.info("Save new best model with dice: {}".format(best_performance))

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % 
                    (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                
                model.train()
                torch.cuda.empty_cache()

            # 10. 定期保存（保持不变）
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