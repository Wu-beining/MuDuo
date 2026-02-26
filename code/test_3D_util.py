

import math
import os
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from tqdm import tqdm

def save_nii(pred_array, reference_itk_img, save_path):
    """
    辅助函数：保存预测结果为 nii.gz
    关键点：必须使用 reference_itk_img 的元数据（原点、方向、层厚），
    否则 ITK-SNAP 里会和原图对不齐。
    """
    # 确保保存的是 uint8 类型 (标签通常是整数)
    pred_img = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    pred_img.CopyInformation(reference_itk_img)  # 核心：拷贝空间信息
    sitk.WriteImage(pred_img, save_path)


def calculate_metric_percase(pred, gt):
    """
    计算单个类别的所有指标: Dice, RAVD, ASD, HD95
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        
        # RAVD (Relative Absolute Volume Difference)
        try:
            ravd = abs(metric.binary.ravd(pred, gt))
        except:
            ravd = 0.0
            
        # ASD (Average Surface Distance)
        try:
            asd = metric.binary.asd(pred, gt)
        except:
            asd = 0.0
            
        # HD95 (Hausdorff Distance 95%)
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except:
            hd95 = 0.0
            
        return np.array([dice, ravd, asd, hd95])
        
    elif pred.sum() == 0 and gt.sum() == 0:
        # 预测空，真值空 -> 完美
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    else:
        # 预测空真值不空，或者预测不空真值空 -> 0分
        return np.array([0.0, 1.0, 0.0, 0.0])


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    """
    滑动窗口预测 (Sliding Window Inference)
    保持原逻辑不变
    """
    w, h, d = image.shape[0], image.shape[1], image.shape[2]
    add_pad = False
    
    # 如果图像尺寸小于 Patch Size，进行 Padding
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

    score_map = torch.zeros((num_classes, ww, hh, dd)).cuda()
    cnt = torch.zeros((ww, hh, dd)).cuda()

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
                
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    
    score_map = score_map / cnt
    label_map = torch.argmax(score_map, dim=0).cpu().numpy()

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        
    return label_map


def test_all_case(net, base_dir, test_list="val.txt", num_classes=2, patch_size=(96, 96, 96), 
                  stride_xy=16, stride_z=16, modality='both', save_result=True, test_save_path=None):
    """
    主测试函数：遍历所有案例，预测并计算指标，可选择保存结果
    """
    # 1. 路径检查与列表加载
    list_path = os.path.join(base_dir, test_list)
    if not os.path.exists(list_path):
        if os.path.exists(os.path.join(os.path.dirname(base_dir), test_list)):
             list_path = os.path.join(os.path.dirname(base_dir), test_list)
    
    with open(list_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]
    
    # 2. 设置保存目录
    if save_result:
        if test_save_path is None:
            # 默认保存在 base_dir 下的 predictions 文件夹
            prediction_dir = os.path.join(base_dir, "predictions")
        else:
            prediction_dir = test_save_path
        
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        print(f"Predictions will be saved to: {prediction_dir}")

    all_metrics = [] 
    print(f"Validation begin: {len(image_list)} cases | Modality: {modality} | Num Classes: {num_classes}")

    for image_name in tqdm(image_list):
        if os.path.exists(os.path.join(base_dir, "images")):
             root_dir = base_dir
        elif os.path.exists(os.path.join(os.path.dirname(base_dir), "images")):
             root_dir = os.path.dirname(base_dir)
        else:
             root_dir = base_dir

        # 3. 读取图像 (并保留 reference_itk 用于保存坐标)
        itk_img = None
        
        if modality == 'ct':
            ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
            itk_img = sitk.ReadImage(ct_path) # 获取参考
            img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img[..., np.newaxis] 
            
        elif modality == 'pet':
            pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
            itk_img = sitk.ReadImage(pet_path) # 获取参考
            img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img[..., np.newaxis] 
            
        else: # modality == 'both'
            ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
            pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
            
            # 通常使用 CT 作为几何参考，因为分辨率和结构更清晰
            itk_img = sitk.ReadImage(ct_path) 
            ct = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            pet = sitk.GetArrayFromImage(sitk.ReadImage(pet_path)).astype(np.float32)
            image = np.stack([ct, pet], axis=-1)

        # 读取 Label
        label_path = os.path.join(root_dir, "labels_ok", f"{image_name}.nii.gz")
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)

        # 4. 执行预测
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        # 5. 保存结果 (如果开启)
        if save_result:
            save_name = os.path.join(prediction_dir, f"{image_name}_pred.nii.gz")
            save_nii(prediction, itk_img, save_name)

        # 6. 计算指标 (针对每个类别)
        case_metric = np.zeros((num_classes-1, 4))
        for i in range(1, num_classes):
            case_metric[i-1, :] = calculate_metric_percase(prediction == i, label == i)
        
        all_metrics.append(case_metric)

    # 7. 汇总所有指标
    all_metrics = np.array(all_metrics) # Shape: (N_cases, num_classes-1, 4)
    mean_metric = np.mean(all_metrics, axis=0)
    std_metric = np.std(all_metrics, axis=0, ddof=0)
    
    return mean_metric, std_metric