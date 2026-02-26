import argparse
import os
import shutil
import math
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from glob import glob
from medpy import metric
from tqdm import tqdm
from collections import defaultdict

from networks.unet_3D import unet_3D

os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
# ==================== 配置 ====================

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/cyf/codes/SSL4MIS/data/autopet', 
                    help='数据根目录')
parser.add_argument('--exp', type=str,
                    default='autopet_ours_20', 
                    help='实验名称')
parser.add_argument('--model', type=str,
                    default='unet_3D', 
                    help='模型名称')
parser.add_argument('--modality', type=str, default='pet',
                    choices=['both', 'ct', 'pet'], 
                    help='输入模态')
parser.add_argument('--model_pool', type=str, nargs='+',
                    default=None,
                    help='模型池路径列表，如: path/to/model1.pth path/to/model2.pth ...')
parser.add_argument('--model_dir', type=str,
                    default=None,
                    help='或指定包含多个模型的目录，自动加载所有*_best_model.pth')
parser.add_argument('--output_oracle_info', type=str,
                    default='oracle_selection_info.json',
                    help='输出每个case选择的模型信息')


# ==================== 核心工具函数 ====================

def save_nii(pred_array, reference_itk_img, save_path):
    """保存预测结果，保持原始空间信息"""
    pred_img = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    pred_img.CopyInformation(reference_itk_img)
    sitk.WriteImage(pred_img, save_path)


def calculate_metric_percase(pred, gt):
    """
    计算单个类别的所有指标: Dice, RAVD, ASD, HD95
    返回: [dice, ravd, asd, hd95]
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        try:
            ravd = abs(metric.binary.ravd(pred, gt))
        except:
            ravd = 0.0
        try:
            asd = metric.binary.asd(pred, gt)
        except:
            asd = 0.0
        try:
            hd95 = metric.binary.hd95(pred, gt)
        except:
            hd95 = 0.0
        return np.array([dice, ravd, asd, hd95])
        
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        return np.array([0.0, 1.0, 0.0, 0.0])


def calculate_case_dice(pred, label, num_classes):
    """
    计算整个case的平均Dice（所有器官的平均）
    这是Oracle选择的核心指标
    """
    case_dice_per_class = []
    for c in range(1, num_classes):
        dice = metric.binary.dc(pred == c, label == c)
        case_dice_per_class.append(dice)
    
    # 返回平均Dice和所有器官的Dice列表
    mean_dice = np.mean(case_dice_per_class)
    return mean_dice, case_dice_per_class


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    """
    滑动窗口预测，返回预测标签和概率图
    """
    w, h, d = image.shape[0], image.shape[1], image.shape[2]
    add_pad = False
    
    # Padding处理
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
        
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad), (0, 0)], 
                      mode='constant', constant_values=0)
    
    ww, hh, dd = image.shape[0], image.shape[1], image.shape[2]
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    # 累积概率图和计数器
    score_map = torch.zeros((num_classes, ww, hh, dd)).cuda()
    cnt = torch.zeros((ww, hh, dd)).cuda()

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                
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
        
    return label_map, score_map


# ==================== Oracle核心：为每个case选最好模型 ====================

class OracleEnsembleInference:
    """
    Oracle集成推理：为每个样本遍历所有模型，用GT选最好的
    """
    def __init__(self, model_paths, num_classes, in_channels, device='cuda'):
        self.model_paths = model_paths
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.device = device
        self.models = []
        self.model_names = []
        
        # 加载所有模型
        for idx, path in enumerate(model_paths):
            net = unet_3D(n_classes=num_classes, in_channels=in_channels).to(device)
            net.load_state_dict(torch.load(path, map_location=device))
            net.eval()
            self.models.append(net)
            
            # 提取模型名称用于记录
            name = os.path.basename(os.path.dirname(path)) if os.path.dirname(path) else f"model_{idx}"
            self.model_names.append(name)
            print(f"[Oracle] 加载模型 {idx}: {name}")
            print(f"         路径: {path}")
        
        self.n_models = len(self.models)
        print(f"\n[Oracle] 共加载 {self.n_models} 个模型")
        
        # 记录每个case的选择信息
        self.selection_log = []
        
    def find_best_model_for_case(self, image, label, stride_xy, stride_z, patch_size, case_name):
        """
        核心：遍历所有模型，用GT计算Dice，选最好的
        
        返回: best_prediction, best_model_idx, best_dice, all_results
        """
        all_predictions = []
        all_dices = []
        all_per_class_dices = []
        
        print(f"\n[Oracle] 处理案例: {case_name}")
        print("-" * 60)
        
        for idx, net in enumerate(self.models):
            # 预测
            with torch.no_grad():
                pred, _ = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, self.num_classes
                )
            
            # 计算Dice
            mean_dice, per_class_dice = calculate_case_dice(pred, label, self.num_classes)
            
            all_predictions.append(pred)
            all_dices.append(mean_dice)
            all_per_class_dices.append(per_class_dice)
            
            print(f"  模型 {idx} ({self.model_names[idx]}): "
                  f"Mean Dice = {mean_dice:.4f}, "
                  f"Per-class = {[f'{d:.3f}' for d in per_class_dice]}")
        
        # 找最好的模型
        best_idx = int(np.argmax(all_dices))
        best_dice = all_dices[best_idx]
        best_pred = all_predictions[best_idx]
        
        print(f"\n  ★★★ 最佳选择: 模型 {best_idx} ({self.model_names[best_idx]})")
        print(f"      Mean Dice: {best_dice:.4f}")
        print(f"      相比最差提升: {best_dice - min(all_dices):.4f}")
        print(f"      相比平均提升: {best_dice - np.mean(all_dices):.4f}")
        
        # 记录选择信息
        selection_info = {
            'case_name': case_name,
            'best_model_idx': best_idx,
            'best_model_name': self.model_names[best_idx],
            'best_dice': float(best_dice),
            'all_model_dices': [float(d) for d in all_dices],
            'all_model_names': self.model_names,
            'dice_gap_best_worst': float(best_dice - min(all_dices)),
            'dice_gap_best_mean': float(best_dice - np.mean(all_dices)),
            'per_class_dices_best': [float(d) for d in all_per_class_dices[best_idx]]
        }
        self.selection_log.append(selection_info)
        
        return best_pred, best_idx, best_dice, selection_info
    
    def test_all_case_oracle(self, base_dir, test_list, patch_size, 
                              stride_xy, stride_z, modality,
                              save_result=True, test_save_path=None):
        """
        Oracle批量推理：每个case独立选最好模型
        """
        # 加载测试列表
        list_path = os.path.join(base_dir, test_list)
        if not os.path.exists(list_path):
            alt_path = os.path.join(os.path.dirname(base_dir), test_list)
            if os.path.exists(alt_path):
                list_path = alt_path
        
        with open(list_path, 'r') as f:
            image_list = [line.strip() for line in f.readlines()]
        
        print(f"\n{'='*70}")
        print(f"[Oracle Inference] 共 {len(image_list)} 个案例")
        print(f"模型池大小: {self.n_models}")
        print(f"{'='*70}")
        
        # 设置保存路径
        if save_result:
            if test_save_path is None:
                test_save_path = os.path.join(base_dir, "predictions_oracle")
            os.makedirs(test_save_path, exist_ok=True)
            print(f"预测结果保存至: {test_save_path}")
        
        # 统计信息
        all_metrics = []
        selection_summary = {
            'model_selected_count': defaultdict(int),
            'model_selected_dice_sum': defaultdict(float)
        }
        
        for image_name in tqdm(image_list, desc="Oracle推理"):
            # 确定数据根目录
            if os.path.exists(os.path.join(base_dir, "images")):
                root_dir = base_dir
            elif os.path.exists(os.path.join(os.path.dirname(base_dir), "images")):
                root_dir = os.path.dirname(base_dir)
            else:
                root_dir = base_dir
            
            # 读取图像
            itk_img = None
            if modality == 'ct':
                ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
                itk_img = sitk.ReadImage(ct_path)
                img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
                image = img[..., np.newaxis]
                in_ch = 1
                
            elif modality == 'pet':
                pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
                itk_img = sitk.ReadImage(pet_path)
                img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
                image = img[..., np.newaxis]
                in_ch = 1
                
            else:  # both
                ct_path = os.path.join(root_dir, "images", f"{image_name}_0000.nii.gz")
                pet_path = os.path.join(root_dir, "images", f"{image_name}_0001.nii.gz")
                itk_img = sitk.ReadImage(ct_path)
                ct = sitk.GetArrayFromImage(itk_img).astype(np.float32)
                pet = sitk.GetArrayFromImage(sitk.ReadImage(pet_path)).astype(np.float32)
                image = np.stack([ct, pet], axis=-1)
                in_ch = 2
            
            # 读取GT标签（Oracle需要！）
            label_path = os.path.join(root_dir, "labels_ok", f"{image_name}.nii.gz")
            if not os.path.exists(label_path):
                # 尝试其他标签路径
                label_path = os.path.join(root_dir, "labels", f"{image_name}.nii.gz")
            label = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.uint8)
            
            # ========== Oracle核心：选最好模型 ==========
            best_prediction, best_idx, best_dice, info = self.find_best_model_for_case(
                image, label, stride_xy, stride_z, patch_size, image_name
            )
            
            # 统计
            selection_summary['model_selected_count'][best_idx] += 1
            selection_summary['model_selected_dice_sum'][best_idx] += best_dice
            
            # 保存预测
            if save_result:
                save_name = os.path.join(test_save_path, f"{image_name}_pred.nii.gz")
                save_nii(best_prediction, itk_img, save_name)
            
            # 计算详细指标（所有类别）
            case_metric = np.zeros((self.num_classes-1, 4))
            for i in range(1, self.num_classes):
                case_metric[i-1, :] = calculate_metric_percase(
                    best_prediction == i, label == i
                )
            all_metrics.append(case_metric)
        
        # 汇总
        all_metrics = np.array(all_metrics)
        mean_metric = np.mean(all_metrics, axis=0)
        std_metric = np.std(all_metrics, axis=0, ddof=0)
        
        # 打印Oracle选择统计
        print(f"\n{'='*70}")
        print("[Oracle选择统计]")
        print(f"{'='*70}")
        total_cases = len(image_list)
        for idx in range(self.n_models):
            count = selection_summary['model_selected_count'][idx]
            if count > 0:
                avg_dice = selection_summary['model_selected_dice_sum'][idx] / count
                percentage = count / total_cases * 100
                print(f"模型 {idx} ({self.model_names[idx]}):")
                print(f"  被选中次数: {count}/{total_cases} ({percentage:.1f}%)")
                print(f"  平均Dice (当被选中时): {avg_dice:.4f}")
        
        return mean_metric, std_metric, self.selection_log


# ==================== 主函数 ====================

def Inference_Oracle(FLAGS):
    """
    Oracle集成推理主函数
    """
    # 确定模型池
    if FLAGS.model_pool:
        model_paths = FLAGS.model_pool
    elif FLAGS.model_dir:
        # 自动发现目录下的所有模型
        pattern = os.path.join(FLAGS.model_dir, "**", "*_best_model.pth")
        model_paths = glob(pattern, recursive=True)
        if not model_paths:
            pattern = os.path.join(FLAGS.model_dir, "*.pth")
            model_paths = glob(pattern)
    else:
        # 默认：从exp目录加载多个fold的模型
        snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
        model_paths = []
        for fold in range(5):  # 假设有5个fold
            path = os.path.join(snapshot_path, f'fold{fold}', f'{FLAGS.model}_best_model.pth')
            if os.path.exists(path):
                model_paths.append(path)
        
        if not model_paths:
            # 回退到单个模型（非Oracle模式，警告）
            path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
            if os.path.exists(path):
                model_paths = [path]
                print("警告: 只找到一个模型，Oracle退化为单模型推理")
    
    if len(model_paths) < 2:
        raise ValueError(f"Oracle模式需要至少2个模型，但只找到 {len(model_paths)} 个")
    
    print(f"\n发现 {len(model_paths)} 个模型:")
    for p in model_paths:
        print(f"  - {p}")
    
    # 确定输入通道数
    if FLAGS.modality == 'ct':
        in_chanes = 1
    elif FLAGS.modality == 'pet':
        in_chanes = 1
    elif FLAGS.modality == 'both':
        in_chanes = 2
    
    num_classes = 11  # 根据你的任务调整
    
    # 创建Oracle推理器
    oracle = OracleEnsembleInference(
        model_paths=model_paths,
        num_classes=num_classes,
        in_channels=in_chanes,
        device='cuda'
    )
    
    # 设置保存路径
    test_save_path = "../model/{}/Prediction_Oracle".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    # 执行Oracle推理
    avg_metric, std_metric, selection_log = oracle.test_all_case_oracle(
        base_dir=FLAGS.root_path,
        test_list="val.txt",  # 注意：Oracle需要GT，所以用val.txt而非test.txt
        patch_size=(128, 128, 128),
        stride_xy=64,
        stride_z=64,
        modality=FLAGS.modality,
        save_result=True,
        test_save_path=test_save_path
    )
    
    # 保存选择日志
    import json
    log_path = os.path.join(test_save_path, FLAGS.output_oracle_info)
    with open(log_path, 'w') as f:
        json.dump({
            'model_pool': model_paths,
            'selections': selection_log,
            'summary': {
                'total_cases': len(selection_log),
                'model_usage': {
                    str(idx): {
                        'name': oracle.model_names[idx],
                        'count': int(oracle.selection_log.count([s for s in oracle.selection_log if s['best_model_idx']==idx]))
                    } for idx in range(oracle.n_models)
                }
            }
        }, f, indent=2)
    print(f"\n选择日志已保存: {log_path}")
    
    # 打印最终指标
    print("\n" + "="*70)
    print("Oracle集成推理 - 最终指标")
    print("="*70)
    print(f"Mean Dice:  {avg_metric[:, 0].mean():.4f} ± {std_metric[:, 0].mean():.4f}")
    print(f"Mean RAVD:  {avg_metric[:, 1].mean():.4f} ± {std_metric[:, 1].mean():.4f}")
    print(f"Mean HD95:  {avg_metric[:, 2].mean():.4f} ± {std_metric[:, 2].mean():.4f}")
    print(f"Mean ASD:   {avg_metric[:, 3].mean():.4f} ± {std_metric[:, 3].mean():.4f}")
    print("="*70)
    
    # 各器官详细指标
    organ_names = ['背景'] + [f'器官{i}' for i in range(1, num_classes)]  # 根据实际修改
    print("\n各器官Dice指标:")
    for i in range(num_classes-1):
        print(f"  {organ_names[i+1]:12s}: {avg_metric[i, 0]:.4f} ± {std_metric[i, 0]:.4f}")
    
    return avg_metric, std_metric, selection_log


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    
    # 强制检查：Oracle模式必须在验证集上运行（有GT）
    print("\n" + "!"*70)
    print("!"*70 + "\n")
    
    metric = Inference_Oracle(FLAGS)