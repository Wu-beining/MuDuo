import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import center_of_mass
import torch.nn.functional as F
# 你的项目依赖
from dataloaders.autopet import AutoPETDataset, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.net_factory_3d import net_factory_3d
from utils import losses, ramps
from val_3D import test_all_case
from tensorboardX import SummaryWriter
# 引入我们刚才改好的工具箱
from segment_anything.build_sam3D import sam_model_registry3D
# from networks.sam_med3d import sam_model_registry3D
from Dualsam_plus import semisam_branch

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/cyf/codes/SSL4MIS/data/autopet')
parser.add_argument('--exp', type=str, default='autopet/DualExpert_Top50')
parser.add_argument('--model', type=str, default='unet_3D')
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=4) # 建议 BS >= 4 以便做筛选
parser.add_argument('--labeled_bs', type=int, default=2)
parser.add_argument('--labeled_num', type=int, default=10) # 比如 10% 标签
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--patch_size', type=list, default=[128, 128, 128])
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--modality', type=str, default='both', help='pet/ct/both')
# 权重路径 (请修改为你自己的路径)
parser.add_argument('--ckpt_unet_prompt', type=str, default='/data/cyf/codes/SSL4MIS/model/autopet_pet_fs_20_20/unet_3D/unet_3D_best_model.pth')
parser.add_argument('--ckpt_sam_ct', type=str, default='/data/cyf/codes/SSL4MIS/trained_pth/sam_med3d_turbo.pth')
parser.add_argument('--ckpt_sam_pet', type=str, default='/data/cyf/codes/SSL4MIS/SegAnyPET/seganypet_v1.pth')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--num_points', type=int, default=3, help='number of points')
args = parser.parse_args()

# ===================================================
#  辅助函数：从 Mask 批量计算重心 (用于 UNet -> Point)
# ===================================================
def dice_loss_1(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    return 1 - (2 * intersection + smooth) / (union + smooth)


def get_batch_centroids(mask_batch, device):
    """
    输入: mask_batch (B, 1, D, H, W) 0/1 二值图
    输出: points (B, 1, 3), labels (B, 1)
    """
    B = mask_batch.shape[0]
    points = torch.zeros((B, 1, 3)).to(device)
    labels = torch.zeros((B, 1)).to(device) # 0表示没找到，1表示找到了
    
    mask_np = mask_batch.squeeze(1).cpu().numpy()
    
    for b in range(B):
        if mask_np[b].sum() > 0:
            z, y, x = center_of_mass(mask_np[b])
            points[b, 0, 0] = int(z)
            points[b, 0, 1] = int(y)
            points[b, 0, 2] = int(x)
            labels[b, 0] = 1 # 标记为有效点
        else:
            # 如果 UNet 没找到，这就设为中心点或者保留0，但在 Loss 计算时要 mask 掉
            labels[b, 0] = 0 
            
    return points, labels

def get_batch_multipoints_gpu(mask_batch, num_points=3):
    """
    纯GPU实现，输入输出都在GPU
    mask_batch: [B, 1, D, H, W] GPU tensor
    返回: points [B, num_points, 3], labels [B, num_points] (都在GPU)
    """
    B, _, D, H, W = mask_batch.shape
    device = mask_batch.device
    
    points = torch.zeros((B, num_points, 3), device=device)
    labels = torch.zeros((B, num_points), device=device)
    
    for b in range(B):
        # GPU操作：找到所有前景位置
        mask_b = mask_batch[b, 0]  # [D, H, W]
        foreground = (mask_b > 0).nonzero(as_tuple=False)  # [N, 3] GPU tensor
        
        if foreground.shape[0] == 0:
            continue
        
        k = min(num_points, foreground.shape[0])
        
        # 取前k个（或随机采样）
        selected = foreground[:k]
        points[b, :k] = selected.float()
        labels[b, :k] = 1
        
        # 填充
        if k < num_points:
            points[b, k:] = points[b, k-1:k]
    
    return points, labels




def get_current_consistency_weight(epoch):
    return 0.1 * ramps.sigmoid_rampup(epoch, 200.0)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
# ===================================================
#  训练主函数
# ===================================================
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = 11
    
    # 1. 加载模型: Student (U-Net)
    # 输入通道=2 (CT+PET)
    
    if args.modality == 'ct':
        in_chanes = 1
    elif args.modality == 'pet':
        in_chanes = 1
    elif args.modality == 'both':
        in_chanes = 2
    
    model = net_factory_3d(net_type=args.model, in_chns=in_chanes, class_num=num_classes).cuda()
    model.train()
    
    # 2. 加载冻结的 Teacher 模型们
    print("Loading Frozen Teachers...")
    
    # A. Prompt Generator (UNet, PET-only)
    prompt_gen_unet = net_factory_3d(net_type='unet_3D', in_chns=1, class_num=num_classes).cuda()
    prompt_gen_unet.load_state_dict(torch.load(args.ckpt_unet_prompt))
    prompt_gen_unet.eval()
    
    # B. CT Expert (SAM-Med3D)
    ct_expert = sam_model_registry3D['vit_b_ori'](checkpoint=args.ckpt_sam_ct).cuda()
    ct_expert.eval()
    
    # C. PET Expert (SegAnyPET)
    pet_expert = sam_model_registry3D['vit_b_ori'](checkpoint=args.ckpt_sam_pet).cuda()
    pet_expert.eval()
    
    # 冻结所有 Teacher 参数
    for m in [prompt_gen_unet, ct_expert, pet_expert]:
        for p in m.parameters(): p.requires_grad = False

    # 3. 数据加载
    db_train = AutoPETDataset(base_dir=args.root_path, split='train', 
                              labeled_num=args.labeled_num, 
                              transform=transforms.Compose([RandomRotFlip(), RandomCrop(args.patch_size), ToTensor()]),
                              modality='both') # 必须是 both
    
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    # logging.info("{} iterations per epoch".format(len(trainloader)))
    
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    iter_num = 0
    max_iterations = args.max_iterations
    best_performance = 0.0
    consistency_loss_total = 0.0
    # ==========================
    #  训练循环 Loop
    # ==========================
    while iter_num < max_iterations:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            # [B, 2, D, H, W] -> Channel 0: CT, Channel 1: PET
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            
            # --- 1. Student Forward (全量数据) ---
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1) # [B, 13, D, H, W]

            # --- 2. 监督 Loss (只在 Labeled 数据上算) ---
            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_ce + loss_dice)

            # --- 3. 无标签数据处理 (关键逻辑!) ---
            if args.labeled_bs < args.batch_size:
                # 取出无标签部分
                unlabeled_vol = volume_batch[args.labeled_bs:]    # [U, 2, D, H, W]
                unlabeled_ct  = unlabeled_vol[:, 0:1, ...]        # CT
                unlabeled_pet = unlabeled_vol[:, 1:2, ...]        # PET
                student_pred_unlab = outputs_soft[args.labeled_bs:] # Student 预测
                
                # 初始化 Loss 累加器
                consistency_loss_total = 0.0
                valid_organ_count = 0
                
                with torch.no_grad():
                    unet_logits_all = prompt_gen_unet(unlabeled_pet)
                
                
                
                
                # ==== 按器官循环 (Class 1 到 12) ====
                for c in range(1, num_classes):
                    
                    # ---------------------------
                    # A. PET 流 (UNet -> Point -> SegAnyPET)
                    # ---------------------------
                    with torch.no_grad():
                        # UNet 粗定位
                        unet_mask = (torch.argmax(unet_logits_all, dim=1) == c).unsqueeze(1)
                        # unet_logits = prompt_gen_unet(unlabeled_pet)
                        # unet_mask = torch.argmax(unet_logits, dim=1) == c
                        # unet_mask = unet_mask.unsqueeze(1).float() # [U, 1, D, H, W]
                        
                        # 提取重心 Point
                        # points: [U, 1, 3], labels: [U, 1]
                        if args.num_points == 1:
                            points, valid_labels = get_batch_centroids(unet_mask, device=volume_batch.device)
                        else:
                            points, valid_labels = get_batch_multipoints_gpu(unet_mask, num_points=args.num_points)
                        
                        # SegAnyPET 推理 (得到 P_PET logits)
                        logits_pet = semisam_branch(
                            unlabeled_pet, pet_expert, prompt_mode='point',
                            prompt_points=points, prompt_labels=valid_labels
                        )
                        prob_pet = torch.sigmoid(logits_pet) # [U, 1, D, H, W]

                    # ---------------------------
                    # B. CT 流 (Student Mask -> SAM-Med3D)
                    # ---------------------------
                    with torch.no_grad():
                        # 使用 Student 当前预测作为 Mask Prompt
                        # student_pred_unlab[:, c:c+1] 是当前器官的 soft mask
                        logits_ct = semisam_branch(
                            unlabeled_ct, ct_expert, prompt_mode='mask',
                            prompt_mask=student_pred_unlab[:, c:c+1]
                        )
                        prob_ct = torch.sigmoid(logits_ct) # [U, 1, D, H, W]

                    # ---------------------------
                    # C. IoU 筛选 (Top 50%)
                    # ---------------------------
                    # 只有当 UNet 找到了点 (valid_labels=1) 时，这个样本才有效
                    # 计算 IoU
                    intersection = (prob_ct * prob_pet).sum(dim=(2,3,4))
                    union = (prob_ct + prob_pet).sum(dim=(2,3,4)) - intersection
                    iou_scores = intersection / (union + 1e-8) # [U, 1]
                    
                    # 将 UNet 没找到点的样本 IoU 设为 -1，确保它排在最后
                    # iou_scores[valid_labels == 0] = -1.0
                    has_valid_point = (valid_labels.sum(dim=1, keepdim=True) > 0)  # [B, 1]
                    iou_scores[~has_valid_point] = -1.0
                    
                    # 排序筛选
                    batch_size_unlab = unlabeled_vol.shape[0]
                    # 计算要保留的数量 (50%)
                    k = max(1, batch_size_unlab // 2) 
                    
                    # 获取 Top K 的索引
                    _, topk_indices = torch.topk(iou_scores.squeeze(), k)
                    
                    # ---------------------------
                    # D. 融合与 Loss 计算
                    # ---------------------------
                    # 融合标签 P_fused (简单平均)
                    # p_fused = (prob_ct + prob_pet) / 2.0
                    
                    # # 取出 Top K 的样本计算 MSE Loss
                    # # Student [U, 13, ...] -> 取当前 Channel c
                    # student_organ = student_pred_unlab[:, c:c+1, ...]
                    
                    # # 只选 Top K
                    # loss_c = torch.mean((student_organ[topk_indices] - p_fused[topk_indices]) ** 2)
                    # consistency_loss_total += loss_c
                    # valid_organ_count += 1
                    
                    # 动态融合
                    alpha = torch.rand(prob_ct.shape[0], 1, 1, 1, 1).to(prob_ct.device)
                    prob_mixed = alpha * prob_ct + (1 - alpha) * prob_pet
                    p_fused_hard = (prob_mixed > 0.5).float() 
                    
                    student_organ_topk = student_pred_unlab[topk_indices, c:c+1, ...]
                    p_fused_hard_topk = p_fused_hard[topk_indices]
                    

            
                    loss_c = dice_loss_1(student_organ_topk, p_fused_hard_topk).mean()
                    consistency_loss_total += loss_c
                    valid_organ_count += 1
            
            
            
            
                # 平均 Consistency Loss
                if valid_organ_count > 0:
                    consistency_loss_total /= valid_organ_count
                
                # 动态权重
                consistency_weight = get_current_consistency_weight(iter_num // 150)
                loss = supervised_loss + consistency_weight * consistency_loss_total
            else:
                loss = supervised_loss

            # --- 4. 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
                
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss_total, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
                
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            
            if iter_num > 0 and iter_num % 1000 == 0:
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
    # 配置 Logger 等
    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path): os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    train(args, snapshot_path)