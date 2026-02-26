import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT

class unet_3D_dv(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=2, is_batchnorm=True):
        """
        in_channels: 输入的总通道数 (例如 CT=1, PET=1, 则 in_channels=2)
        """
        super(unet_3D_dv, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # 假设 CT 和 PET 各占一半通道 (例如各 1 个通道)
        self.branch_channels = in_channels // 2
        
        # 基础滤波器数量
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # ==================== CT Encoder ====================
        self.conv1_ct = UnetConv3(self.branch_channels, filters[0], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1_ct = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2_ct = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2_ct = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3_ct = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3_ct = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4_ct = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4_ct = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center_ct = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))

        # ==================== PET Encoder ====================
        self.conv1_pet = UnetConv3(self.branch_channels, filters[0], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1_pet = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2_pet = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2_pet = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3_pet = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3_pet = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4_pet = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4_pet = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center_pet = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))

        # ==================== Decoder (Shared) ====================
        # 注意：因为 Concat 了 CT 和 PET 的特征，所以输入通道数变成了原来的 2 倍
        # filters[i] * 2
        
        # up_concat4 接收: Concat后的Center (filters[4]*2) 和 Concat后的Conv4 (filters[3]*2)
        # 输出目标: filters[3]*2 (为了保持通道数平衡，通常解码器维持双倍宽度，或者在这里降维)
        # 这里我们假设 Decoder 也维持双倍宽度以保留双模态信息
        self.up_concat4 = UnetUp3_CT(filters[4]*2, filters[3]*2, is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3]*2, filters[2]*2, is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2]*2, filters[1]*2, is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1]*2, filters[0]*2, is_batchnorm)

        # final conv
        # 输入是 up_concat1 的输出 (filters[0]*2)
        self.final = nn.Conv3d(filters[0]*2, n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x_ct = inputs[:, :self.branch_channels, ...]
        x_pet = inputs[:, self.branch_channels:, ...]

        conv1_ct = self.conv1_ct(x_ct)
        maxpool1_ct = self.maxpool1_ct(conv1_ct)

        conv2_ct = self.conv2_ct(maxpool1_ct)
        maxpool2_ct = self.maxpool2_ct(conv2_ct)

        conv3_ct = self.conv3_ct(maxpool2_ct)
        maxpool3_ct = self.maxpool3_ct(conv3_ct)

        conv4_ct = self.conv4_ct(maxpool3_ct)
        maxpool4_ct = self.maxpool4_ct(conv4_ct)

        center_ct = self.center_ct(maxpool4_ct)
        center_ct = self.dropout1(center_ct)

        conv1_pet = self.conv1_pet(x_pet)
        maxpool1_pet = self.maxpool1_pet(conv1_pet)

        conv2_pet = self.conv2_pet(maxpool1_pet)
        maxpool2_pet = self.maxpool2_pet(conv2_pet)

        conv3_pet = self.conv3_pet(maxpool2_pet)
        maxpool3_pet = self.maxpool3_pet(conv3_pet)

        conv4_pet = self.conv4_pet(maxpool3_pet)
        maxpool4_pet = self.maxpool4_pet(conv4_pet)

        center_pet = self.center_pet(maxpool4_pet)
        center_pet = self.dropout1(center_pet)

        center_cat = torch.cat([center_ct, center_pet], dim=1)

        conv4_cat = torch.cat([conv4_ct, conv4_pet], dim=1)
        conv3_cat = torch.cat([conv3_ct, conv3_pet], dim=1)
        conv2_cat = torch.cat([conv2_ct, conv2_pet], dim=1)
        conv1_cat = torch.cat([conv1_ct, conv1_pet], dim=1)

        # ----- Decoder -----

        up4 = self.up_concat4(conv4_cat, center_cat) 
        up3 = self.up_concat3(conv3_cat, up4)
        up2 = self.up_concat2(conv2_cat, up3)
        up1 = self.up_concat1(conv1_cat, up2)
        
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final