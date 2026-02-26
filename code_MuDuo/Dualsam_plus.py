import torch
import torch.nn.functional as F
from segment_anything.build_sam3D import sam_model_registry3D

# ===================================================
#  功能函数：不确定性计算
# ===================================================
def compute_epistemic_uncertainty(all_preds):
    predictions = torch.stack(all_preds)
    ensemble = torch.mean(predictions, dim=0)
    uncertainty = torch.mean((predictions - ensemble) ** 2, dim=0)
    return unc

# ===================================================
#  模式 A: 基于外部 Point 的推理 (用于 PET 流 - UNet引导)
# ===================================================
def infer_with_external_points(img3D, points, labels, sam_model, device='cuda'):
    """
    img3D: (B, C, D, H, W)
    points: (B, N, 3) 来自 UNet 的重心
    labels: (B, N) 全为 1
    """
    batch_size = img3D.shape[0]
    
    # 1. Image Encoder (只跑一次)
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img3D.to(device))

    pred_batch = []
    
    for b in range(batch_size):
        # 拿到当前样本的点和图
        # 注意：points 维度需要是 (1, N, 3)
        curr_points = points[b].unsqueeze(0).to(device).float()
        curr_labels = labels[b].unsqueeze(0).to(device).float()
        
        # 2. Prompt Encoder
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=[curr_points, curr_labels],
            boxes=None,
            masks=None,
        )
        
        # 3. Mask Decoder
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding[b:b+1],
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # 4. 上采样回原图
        high_res_mask = F.interpolate(
            low_res_masks, 
            size=img3D.shape[-3:], 
            mode='trilinear', 
            align_corners=False
        )
        pred_batch.append(high_res_mask)

    return torch.cat(pred_batch, dim=0) # (B, 1, D, H, W)

# ===================================================
#  模式 B: 基于 Mask 的推理 (用于 CT 流 - Student引导)
# ===================================================
def infer_with_mask_prompt(img3D, input_mask, sam_model, device='cuda'):
    """
    input_mask: Student 的预测结果作为 Prompt
    """
    batch_size = img3D.shape[0]
    crop_size = 128
    
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img3D.to(device))
        
        # 下采样 mask
        low_res_mask = F.interpolate(input_mask.float(), size=(crop_size // 4, crop_size // 4, crop_size // 4))
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=low_res_mask.to(device)
        )
        
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        high_res_masks = F.interpolate(low_res_masks, size=img3D.shape[-3:], mode='trilinear', align_corners=False)
        
    return high_res_masks

# ===================================================
#  统一入口函数
# ===================================================
def semisam_branch(volume_batch, expert_model, prompt_mode='mask', 
                   prompt_mask=None, prompt_points=None, prompt_labels=None, device='cuda'):
    """
    参数:
        prompt_mode: 'mask' (CT流) 或 'point' (PET流)
        prompt_mask: 当 mode='mask' 时传入
        prompt_points: 当 mode='point' 时传入
    """
    expert_model.eval()
    
    if prompt_mode == 'mask':
        logits = infer_with_mask_prompt(volume_batch, prompt_mask, expert_model, device)
        
    elif prompt_mode == 'point':
        logits = infer_with_external_points(volume_batch, prompt_points, prompt_labels, expert_model, device)
    
    # 返回 Logits (还没 Sigmoid)
    return logits