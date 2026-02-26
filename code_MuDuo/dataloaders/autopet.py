
# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import itertools
# from torch.utils.data.sampler import Sampler
# import SimpleITK as sitk

# class AutoPETDataset(Dataset):
#     """ 
#     AutoPET Dataset 
#     - 支持多模态选择 (modality)
#     - 支持自定义划分列表 (list_name)
#     """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None, modality='both', labeled_num=5):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []
#         self.modality = modality
        
#         # ================== 【核心修改逻辑】 ==================
#         # 1. 如果传入了具体的 list_name (例如 'splits/train_5.txt')，直接用它
        
#         list_dir = os.path.join(self._base_dir, f'lists_{labeled_num}')
#         txt_path = os.path.join(list_dir, f'{split}.txt') # split 是 'train', 'val' 或 'test'
        
#         with open(txt_path, 'r') as f:
#             self.image_list = [line.strip() for line in f.readlines()]

#         print(f"AutoPET Dataset | Path: {txt_path} | Total: {len(self.image_list)} | Modality: {self.modality}")

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
        
#         # 1. 定义路径
#         # 假设数据在 base_dir/images 下
        
#         ct_path = os.path.join(self._base_dir, "images", "{}_0000.nii.gz".format(image_name))
#         pet_path = os.path.join(self._base_dir, "images", "{}_0001.nii.gz".format(image_name))
#         # label_path = os.path.join(self._base_dir, "labels", "{}.nii.gz".format(image_name))
#         label_path = os.path.join(self._base_dir, "labels", "{}.nii.gz".format(image_name))


#         if os.path.exists(label_path):
#             # 1. 有标签数据：正常读取
#             itk_label = sitk.ReadImage(label_path)
#             label_arr = sitk.GetArrayFromImage(itk_label).astype(np.uint8)
#         else:
#             # 2. 无标签数据：生成全0的 Dummy Label，尺寸与图像相同
#             # 获取图像的空间维度 (D, H, W)
#             d, h, w = image.shape[:3] 
#             label_arr = np.zeros((d, h, w), dtype=np.uint8)

#         # 2. 根据 modality 读取数据
#         if self.modality == 'ct':
#             # 只读 CT
#             itk_img = sitk.ReadImage(ct_path)
#             img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
#             # 增加通道维: (D, H, W) -> (D, H, W, 1)
#             image = img_arr[..., np.newaxis]
            
#         elif self.modality == 'pet':
#             # 只读 PET
#             itk_img = sitk.ReadImage(pet_path)
#             img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
#             # 增加通道维: (D, H, W) -> (D, H, W, 1)
#             image = img_arr[..., np.newaxis]
            
#         else: # 'both'
#             # 读 CT 和 PET 并堆叠
#             itk_ct = sitk.ReadImage(ct_path)
#             itk_pet = sitk.ReadImage(pet_path)
#             ct_arr = sitk.GetArrayFromImage(itk_ct).astype(np.float32)
#             pet_arr = sitk.GetArrayFromImage(itk_pet).astype(np.float32)
#             # 堆叠: (D, H, W, 2)
#             image = np.stack([ct_arr, pet_arr], axis=-1)

#         # 读取 Label
#         itk_label = sitk.ReadImage(label_path)
#         label_arr = sitk.GetArrayFromImage(itk_label).astype(np.uint8)
        
#         sample = {'image': image, 'label': label_arr}
        
#         if self.transform:
#             sample = self.transform(sample)
            
#         return sample

# # --- 以下 Transforms 保持不变 ---
# class CenterCrop(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd), (0, 0)], mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

#         (d, h, w, c) = image.shape
#         d1 = int(round((d - self.output_size[0]) / 2.))
#         h1 = int(round((h - self.output_size[1]) / 2.))
#         w1 = int(round((w - self.output_size[2]) / 2.))

#         label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
#         image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2], :]
#         return {'image': image, 'label': label}

# class RandomCrop(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd), (0, 0)], mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

#         (d, h, w, c) = image.shape
#         d1 = np.random.randint(0, d - self.output_size[0])
#         h1 = np.random.randint(0, h - self.output_size[1])
#         w1 = np.random.randint(0, w - self.output_size[2])

#         label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
#         image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2], :]
#         return {'image': image, 'label': label}

# class RandomRotFlip(object):
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         k = np.random.randint(0, 4)
#         image = np.rot90(image, k, axes=(1, 2)).copy() 
#         label = np.rot90(label, k, axes=(1, 2)).copy()
#         axis = np.random.randint(0, 3)
#         image = np.flip(image, axis=axis).copy()
#         label = np.flip(label, axis=axis).copy()
#         return {'image': image, 'label': label}

# class ToTensor(object):
#     def __call__(self, sample):
#         image = sample['image']
#         # (D, H, W, C) -> (C, D, H, W)
#         image = image.transpose((3, 0, 1, 2)).astype(np.float32)
#         return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
    

# # --- Sampler 部分 (半监督训练必备) ---
# class TwoStreamBatchSampler(Sampler):
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size

#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                    grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size

# def iterate_once(iterable):
#     return np.random.permutation(iterable)

# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#     return itertools.chain.from_iterable(infinite_shuffles())

# def grouper(iterable, n):
#     args = [iter(iterable)] * n
#     return zip(*args)


import os
import torch
import numpy as np
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk

class AutoPETDataset(Dataset):
    """ 
    AutoPET Dataset (3D 半监督适用)
    - 支持多模态选择 (modality: 'ct' | 'pet' | 'both')
    - 自动适配不同 labeled_num 的数据列表
    - 自动生成无标签数据的 Dummy Label (防闪退)
    """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, modality='both', labeled_num=5):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.modality = modality 
        
        # ================== 【动态路径定位】 ==================
        # 自动定位到对应的子文件夹，例如: /data/autopet/lists_5/train.txt
        list_dir = os.path.join(self._base_dir, f'lists_{labeled_num}')
        txt_path = os.path.join(list_dir, f'{split}.txt') 
        # ====================================================

        with open(txt_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.image_list = self.image_list[:num]
            
        print(f"AutoPET Dataset [{split}] | Path: {txt_path} | Total: {len(self.image_list)} | Modality: {self.modality}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        
        # 1. 定义 CT 和 PET 的绝对路径
        ct_path = os.path.join(self._base_dir, "images", "{}_0000.nii.gz".format(image_name))
        pet_path = os.path.join(self._base_dir, "images", "{}_0001.nii.gz".format(image_name))

        # 2. 读取图像并根据 modality 处理通道
        if self.modality == 'ct':
            itk_img = sitk.ReadImage(ct_path)
            img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img_arr[..., np.newaxis] # (D, H, W, 1)
            
        elif self.modality == 'pet':
            itk_img = sitk.ReadImage(pet_path)
            img_arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
            image = img_arr[..., np.newaxis] # (D, H, W, 1)
            
        else: # 'both' 模式 (双通道融合)
            itk_ct = sitk.ReadImage(ct_path)
            itk_pet = sitk.ReadImage(pet_path)
            ct_arr = sitk.GetArrayFromImage(itk_ct).astype(np.float32)
            pet_arr = sitk.GetArrayFromImage(itk_pet).astype(np.float32)
            image = np.stack([ct_arr, pet_arr], axis=-1) # (D, H, W, 2)

        # 3. 读取 Label (关键防崩溃修复区)
        label_path = os.path.join(self._base_dir, "labels_ok", "{}.nii.gz".format(image_name))
        
        if os.path.exists(label_path):
            # 真实有标签数据：正常读取
            itk_label = sitk.ReadImage(label_path)
            label_arr = sitk.GetArrayFromImage(itk_label).astype(np.uint8)
        else:
            # 无标签数据：生成与图像尺寸一致的全 0 假标签
            d, h, w = image.shape[:3]
            label_arr = np.zeros((d, h, w), dtype=np.uint8)
        
        sample = {'image': image, 'label': label_arr}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# ================== 【3D Transforms】 ==================
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd), (0, 0)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (d, h, w, c) = image.shape
        d1 = int(round((d - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        w1 = int(round((w - self.output_size[2]) / 2.))

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2], :]
        return {'image': image, 'label': label}

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd), (0, 0)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (d, h, w, c) = image.shape
        d1 = np.random.randint(0, d - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        w1 = np.random.randint(0, w - self.output_size[2])

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2], :]
        return {'image': image, 'label': label}
    

    
    

class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2)).copy() 
        label = np.rot90(label, k, axes=(1, 2)).copy()
        axis = np.random.randint(0, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        # 维度转换: (D, H, W, C) -> (C, D, H, W)
        image = image.transpose((3, 0, 1, 2)).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
    

# ================== 【半监督 Sampler】 ==================
class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)