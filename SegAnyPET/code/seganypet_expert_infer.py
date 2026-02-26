import torch
import numpy as np
import SimpleITK as sitk
import copy
import os

# 导入 SegAnyPET 工具
from segment_anything import sam_model_registry3D
from utils.infer_utils import (
    get_subject_and_meta_info, 
    data_preprocess, 
    sam_model_infer, 
    data_postprocess, 
    save_numpy_to_nifti,
    read_arr_from_nifti 
)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 12 类器官对应索引 (根据 AutoPET-Organ 数据集)
ORGAN_LIST = {
    1: "liver", 2: "left_kidney", 3: "right_kidney", 4: "heart", 
    5: "spleen", 6: "aorta", 7: "prostate", 8: "left_lung_lower", 
    9: "right_lung_lower", 10: "left_lung_upper", 11: "right_lung_upper", 
    12: "right_lung_middle"
}

def seganypet_expert_inference(image_path, checkpoint_path, device="cuda"):
    """
    SegAnyPET 专家推理主函数
    返回: 
        final_hard_mask: (H, W, D) 离散标签图 (1-12)
        prob_maps: {class_id: prob_array} 概率图字典，用于后期 Fusion
    """
    # 1. 初始化模型
    model = sam_model_registry3D["vit_b_ori"](checkpoint=checkpoint_path)
    model.to(device).eval()

    # 2. 准备基础信息
    # 注意：无标签数据推理时，我们将 image_path 同时传给 gt 参数作为占位
    subject, meta_info = get_subject_and_meta_info(image_path, image_path)
    img_arr, _ = read_arr_from_nifti(image_path, get_meta_info=True)
    
    # 准备存储容器
    final_hard_mask = np.zeros(img_arr.shape, dtype=np.uint8)
    prob_maps = {} # 存储 12 类的软标签 P_PET

    # 3. 循环推理 12 类器官
    for class_id, organ_name in ORGAN_LIST.items():
        print(f"Expert Predicting: {organ_name}...")
        
        # 拷贝元信息，防止处理干扰
        curr_subject = copy.deepcopy(subject)
        curr_meta = copy.deepcopy(meta_info)


        roi_image, _, class_meta = data_preprocess(
            curr_subject, curr_meta, 
            category_index=1, 
            target_spacing=(1.5, 1.5, 1.5), 
            crop_size=128
        )


        with torch.no_grad():

            res_mask, res_prob, _ = sam_model_infer(
                model, roi_image, roi_gt=None, num_clicks=1
            )

        hard_mask_full = data_postprocess(res_mask, class_meta)
        prob_map_full = data_postprocess(res_prob, class_meta) # 修改 data_postprocess 支持 float 输入


        final_hard_mask[hard_mask_full > 0] = class_id
        prob_maps[class_id] = prob_map_full

    return final_hard_mask, prob_maps, meta_info

# ---------------------------------------------------------
# 针对你论文中 IoU 筛选和 Fusion 的逻辑演示
# ---------------------------------------------------------
def process_unlabeled_data(image_pet, image_ct, seg_pet_expert, sam_med3d_expert):
    # 1. 生成 P_PET (软标签 + 硬掩码)
    hard_pet, soft_pet, meta = seganypet_expert_inference(image_pet, "/data/cyf/codes/SSL4MIS/SegAnyPET-main/seganypet_v1.pth")
    
    # 2. 生成 P_CT (通过 SAM-Med3D)
    # hard_ct, soft_ct = sam_med3d_expert(image_ct)
    
    # 3. 计算 IoU 相似度 (用于置信度排序)
    # iou_score = compute_iou(hard_pet, hard_ct)
    
    # 4. 加权融合生成 P_fused (软标签融合)
    # p_fused = (soft_pet * w1 + soft_ct * w2)
    
    return hard_pet, soft_pet, meta


if __name__ == '__main__':
    image_pet = "/data/cyf/codes/SSL4MIS/SegAnyPET-main/fdg_0b57b247b6_05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966_0001.nii.gz"
    image_ct = "/data/cyf/codes/SSL4MIS/SegAnyPET-main/fdg_0b57b247b6_05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966_0000.nii.gz"
    seg_pet_expert = "data/autopet/test/00001/seganypet.nii.gz"
    sam_med3d_expert = "data/autopet/test/00001/sam_med3d.nii.gz"
    seg_pet_hard_path = "/data/cyf/codes/SSL4MIS/SegAnyPET-main/seganypet_hard.nii.gz"
    seg_pet_soft_path = "/data/cyf/codes/SSL4MIS/SegAnyPET-main/seganypet_soft.nii.gz"
    os.makedirs(os.path.dirname(seg_pet_hard_path), exist_ok=True)
    os.makedirs(os.path.dirname(seg_pet_soft_path), exist_ok=True)
    hard_pet, soft_pet, meta = process_unlabeled_data(
        image_pet, image_ct,None,None)
    
    save_numpy_to_nifti(hard_pet, seg_pet_hard_path, meta)
    print(f"Hard mask saved to: {seg_pet_hard_path}")


    soft_fusion = np.zeros_like(hard_pet, dtype=np.float32)
    for cid in soft_pet:
        soft_fusion = np.maximum(soft_fusion, soft_pet[cid])
    

    save_numpy_to_nifti(soft_fusion, seg_pet_soft_path, meta)
    print(f"Soft mask saved to: {seg_pet_soft_path}")
