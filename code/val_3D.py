

import math
import os
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from tqdm import tqdm
import torch.nn.functional as F
def test_all_case(net, base_dir, test_list="val.txt", num_classes=13, patch_size=(96, 96, 96), stride_xy=16, stride_z=16, modality='both'):
    # 路径检查
    list_path = os.path.join(base_dir, test_list)
    if not os.path.exists(list_path):
        # 兼容性寻找
        if os.path.exists(os.path.join(os.path.dirname(base_dir), test_list)):
             list_path = os.path.join(os.path.dirname(base_dir), test_list)
    
    with open(list_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    
    total_metric = np.zeros((num_classes-1, 2))
    print(f"Validation begin: {len(image_list)} cases | Modality: {modality}")

    for image_name in tqdm(image_list):
        # 路径适配
        if os.path.exists(os.path.join(base_dir, "images")):
             root_dir = base_dir
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), "images")):
             root_dir = os.path.dirname(base_dir)
        else:
             root_dir = base_dir

        # ================= 按需读取 =================
        if modality == 'ct':
            ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
            itk_img = sitk.ReadImage(ct_path)
            img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img[..., np.newaxis] # (D, H, W, 1)

        elif modality == 'pet':
            pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
            itk_img = sitk.ReadImage(pet_path)
            img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img[..., np.newaxis] # (D, H, W, 1)

        else: # both
            ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
            pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
            ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
            pet = sitk.GetArrayFromImage(sitk.ReadImage(pet_path)).astype(np.float32)
            image = np.stack([ct, pet], axis=-1) # (D, H, W, 2)
        # ===========================================

        label_path = os.path.join(root_dir, "labels_ok", f"{image_name}.nii.gz")
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)

        # 预测
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if prediction.shape != label.shape:
            # 只有当 PET 尺寸和 Label 尺寸不同时触发 (例如 316x341 vs 400x400)
            print(f"⚠️ 触发自动对齐 | 预测尺寸 {prediction.shape} -> 标签尺寸 {label.shape}")
            
            # 转为 Tensor 并增加 batch 和 channel 维度: (1, 1, D, H, W)
            pred_tensor = torch.from_numpy(prediction.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            # 使用 Nearest (最近邻插值) 将预测结果缩放到 Label 的真实大小
            # 注意：分割 Mask 只能用 'nearest'，绝不能用 'bilinear' 否则类别会变成小数
            pred_resized = F.interpolate(pred_tensor, size=label.shape, mode='nearest')
            
            # 还原为 numpy array
            prediction = pred_resized.squeeze(0).squeeze(0).numpy().astype(np.uint8)


        if np.sum(prediction) > 0:
            for i in range(1, num_classes):
                total_metric[i-1, :] += cal_metric(label == i, prediction == i)
        else:
            # 如果预测全空，Dice设为0 (除非Label也全空，这由cal_metric处理)
             for i in range(1, num_classes):
                total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    return total_metric / len(image_list)

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except:
            hd95 = 0.0
        return np.array([dice, hd95])
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 0.0])

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape[0], image.shape[1], image.shape[2]
    # Padding
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2

    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad), (0, 0)], mode='constant', constant_values=0)
    
    ww, hh, dd = image.shape[0], image.shape[1], image.shape[2]
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    # score_map = torch.zeros((num_classes, ww, hh, dd)).cuda()
    # cnt = torch.zeros((ww, hh, dd)).cuda()
    
    score_map = torch.zeros((num_classes, ww, hh, dd)).cpu() 
    cnt = torch.zeros((ww, hh, dd)).cpu()

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2], :]
                test_patch = np.expand_dims(np.transpose(test_patch, (3, 0, 1, 2)), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    y = torch.softmax(y1, dim=1)
                # y = y[0, :, :, :, :]
                # score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y[0, :, :, :, :].cpu()
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y[0, :, :, :, :].cpu()
                # score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    
    score_map = score_map / cnt
    label_map = torch.argmax(score_map, dim=0).cpu().numpy()

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map