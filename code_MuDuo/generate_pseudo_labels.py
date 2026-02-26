import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import torch.nn.functional as F

# 导入你的模型定义
# from networks.net_factory_3d import net_factory_3d  # UNet
from networks.unet_3D import unet_3D
from SegAnyPET.code.segment_anything import sam_model_registry3D as seganypet_registry

from networks.seganypet import sam_model_registry3D as seganypet_registry # PET Expert

# 12 类定义
ORGAN_LIST = {
    1: "liver", 2: "left_kidney", 3: "right_kidney", 4: "heart", 
    5: "spleen", 6: "aorta", 7: "prostate", 8: "left_lung_lower", 
    9: "right_lung_lower", 10: "left_lung_upper", 11: "right_lung_upper", 
    12: "right_lung_middle"
}

def get_centroid(mask):
    """计算二值 Mask 的重心，返回 [1, 1, 3] 的 Tensor"""
    if np.sum(mask) > 0:
        z, y, x = center_of_mass(mask)
        # 注意坐标顺序，确保与你的 Data Loader 一致 (通常是 D, H, W)
        return torch.tensor([[[int(z), int(y), int(x)]]])
    return None

def run_sam_inference(model, image_embedding, prompt_point, original_size):
    """通用 SAM 推理函数 (适用于 SAM-Med3D 和 SegAnyPET)"""
    device = image_embedding.device
    
    # 构造 Prompt
    points = prompt_point.to(device).float()
    labels = torch.tensor([[1]]).to(device) # 1 = 正点
    
    # Prompt Encoder
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=[points, labels],
        boxes=None,
        masks=None,
    )
    
    # Mask Decoder
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False, # 单点推理，False 是对的
    )
    
    # 插值回原图尺寸
    hr_mask = F.interpolate(
        low_res_masks,
        size=original_size,
        mode='trilinear',
        align_corners=False
    )
    return (torch.sigmoid(hr_mask) > 0.5).squeeze().cpu().numpy()

def generate_pseudo_labels(pet_path, ct_path, unet_model, ct_teacher, pet_teacher, device='cuda'):
    """
    核心流程：UNet 找点 -> CT/PET 专家分别分割 -> 返回结果
    """
    # 1. 读取数据
    itk_pet = sitk.ReadImage(pet_path)
    itk_ct = sitk.ReadImage(ct_path)
    
    pet_arr = sitk.GetArrayFromImage(itk_pet).astype(np.float32)
    ct_arr = sitk.GetArrayFromImage(itk_ct).astype(np.float32)
    
    # 预处理 (Z-Score 等，需与你训练时一致)
    # 这里假设已经预处理好，直接转 Tensor [1, 1, D, H, W]
    pet_tensor = torch.from_numpy(pet_arr).unsqueeze(0).unsqueeze(0).to(device)
    ct_tensor = torch.from_numpy(ct_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    D, H, W = pet_arr.shape
    
    # 2. 【UNet 向导】生成粗糙 Mask
    print("  Step 1: UNet Generating Guidance...")
    unet_model.eval()
    with torch.no_grad():
        # 假设 UNet 输入是 PET (或双模态，按你训练的来)
        unet_input = pet_tensor # 如果 UNet 是双模态训练的，这里要 cat
        unet_out = unet_model(unet_input)
        coarse_mask_all = torch.argmax(unet_out, dim=1).squeeze().cpu().numpy() # [D, H, W]

    # 3. 【Experts 准备】预计算 Image Embeddings (加速)
    print("  Step 2: Experts Encoding Images...")
    ct_teacher.eval()
    pet_teacher.eval()
    with torch.no_grad():
        ct_embedding = ct_teacher.image_encoder(ct_tensor)
        pet_embedding = pet_teacher.image_encoder(pet_tensor)

    # 结果容器
    mask_ct_final = np.zeros((D, H, W), dtype=np.uint8)
    mask_pet_final = np.zeros((D, H, W), dtype=np.uint8)
    
    # 4. 【循环推理】针对 12 个器官
    print("  Step 3: Iterative Inference...")
    for class_id, organ_name in ORGAN_LIST.items():
        # A. 从 UNet 结果提取中心点
        coarse_organ = (coarse_mask_all == class_id)
        prompt_point = get_centroid(coarse_organ)
        
        if prompt_point is None:
            # print(f"    - {organ_name} (Class {class_id}): UNet Missed. Skipping.")
            continue
        
        # B. CT 专家推理
        with torch.no_grad():
            binary_mask_ct = run_sam_inference(ct_teacher, ct_embedding, prompt_point, (D, H, W))
            mask_ct_final[binary_mask_ct == 1] = class_id # 填入结果
            
        # C. PET 专家推理
        with torch.no_grad():
            binary_mask_pet = run_sam_inference(pet_teacher, pet_embedding, prompt_point, (D, H, W))
            mask_pet_final[binary_mask_pet == 1] = class_id # 填入结果
            
    return mask_ct_final, mask_pet_final, itk_pet

# --- 主程序入口 ---
if __name__ == "__main__":
    device = "cuda"
    
    # 1. 加载三个模型
    num_classes = 13
    in_chanes = 1
    print("Loading Models...")
    # A. UNet (向导)
    unet = unet_3D(n_classes=num_classes, in_channels=in_chanes).cuda()
    # unet = net_factory_3d(net_type="unet_3D", in_chns=1, class_num=13).to(device)
    unet.load_state_dict(torch.load("/data/cyf/codes/SSL4MIS/model/autopet_pet_fs_10_10/unet_3D/unet_3D_best_model.pth")) # <--- 改路径
    
    # B. SAM-Med3D (CT 专家)
    ct_model = sam_med3d_registry["vit_b_ori"](checkpoint="/data/cyf/codes/SSL4MIS/trained_pth/sam_med3d_turbo.pth").to(device)
    
    # C. SegAnyPET (PET 专家)
    pet_model = seganypet_registry["vit_b_ori"](checkpoint="/data/cyf/codes/SSL4MIS/SegAnyPET/seganypet_v1.pth").to(device)
    
    # 2. 设置数据路径
    pet_file = "/data/cyf/codes/SSL4MIS/PETCT_0af7ffe12a_08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698__0001.nii.gz"
    ct_file = "/data/cyf/codes/SSL4MIS/PETCT_0af7ffe12a_08-12-2005-NA-PET-CT Ganzkoerper  primaer mit KM-96698__0000.nii.gz"
    
    # 3. 运行生成
    print(f"Processing...")
    p_ct, p_pet, ref_itk = generate_pseudo_labels(pet_file, ct_file, unet, ct_model, pet_model)
    
    # 4. 保存结果 (用于后续 Fusion 或查看)
    def save_nii(arr, ref, name):
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        sitk.WriteImage(img, name)
        
    save_nii(p_ct, ref_itk, "Pseudo_CT_Expert.nii.gz")
    save_nii(p_pet, ref_itk, "Pseudo_PET_Expert.nii.gz")
    
    print("Done! Pseudo labels saved.")