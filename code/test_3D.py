import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from test_3D_util import test_all_case
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/cyf/codes/SSL4MIS/data/autopet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='autopet_pet_fs_5', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')
parser.add_argument('--modality', type=str, default='pet', 
                    choices=['both', 'ct', 'pet'], help='input modality')

def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 11
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    if FLAGS.modality == 'ct':
        in_chanes = 1
    elif FLAGS.modality == 'pet':
        in_chanes = 1
    elif FLAGS.modality == 'both':
        in_chanes = 2
    
    
    net = unet_3D(n_classes=num_classes, in_channels=in_chanes).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    # avg_metric = test_all_case(net, base_dir=FLAGS.root_path, test_list="test.txt", 
    #                            num_classes=num_classes,
    #                            patch_size=(96, 96, 96), stride_xy=64, 
    #                            stride_z=64,modality=FLAGS.modality)
    avg_metric, std_metric = test_all_case(
    net,                      # 你的网络模型
    base_dir=FLAGS.root_path,    # 数据集根目录
    test_list="val.txt",        # 验证列表文件名
    num_classes=num_classes,             # 【重要】如果是12个器官+背景，这里要设为 13
    patch_size=(128, 128, 128), # 例如 [96, 96, 96]
    stride_xy=64, 
    stride_z=64,
    modality=FLAGS.modality,     # 'both', 'ct', 或 'pet'
    save_result=True,           # 开启保存！
    test_save_path=test_save_path # 指定保存位置
)
    

    
    
    
    print("\n" + "="*60)
    print("详细指标说明：")
    print("第1列: Dice (相似度)")
    print("第2列: RAVD (体积差异率)")  
    print("第3列: HD95 (95%边界距离 mm)")
    print("第4列: ASD (平均表面距离 mm)")
    print("="*60)
    return avg_metric, std_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print(f"Mean Dice: {metric[0][:, 0].mean()}")
    print(f"Mean RAVD: {metric[0][:, 1].mean()}")
    print(f"Mean HD95: {metric[0][:, 2].mean()}")
    print(f"Mean ASD: {metric[0][:, 3].mean()}")
    print(f"STD Dice: {metric[1][:, 0].mean()}")
    print(f"STD RAVD: {metric[1][:, 1].mean()}")
    print(f"STD HD95: {metric[1][:, 2].mean()}")
    print(f"STD ASD: {metric[1][:, 3].mean()}")
