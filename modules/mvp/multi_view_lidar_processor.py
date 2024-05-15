from modules.mvp.multi_view_processor import MultiViewProcessor
from mmengine.config import Config
from mmdet3d.apis import init_model
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # If the input and output dimensions don't match, use a 1x1 convolution
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class PseudoImageCNN(nn.Module):
    def __init__(self):
        super(PseudoImageCNN, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.residual_block1 = ResidualBlock(128, 128)
        self.residual_block2 = ResidualBlock(128, 128)
        self.residual_block3 = ResidualBlock(128, 256, stride=2)  # Downsample
        self.residual_block4 = ResidualBlock(256, 256)
        self.residual_block5 = ResidualBlock(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(256, 768)  # Fully connected layer to output 768-dimensional feature vector

    def forward(self, x):
        # Input shape: N x 64 x 400 x 400
        out = self.l1(x)  # N x 128 x 400 x 400
        out = self.residual_block1(out)  # N x 128 x 400 x 400
        out = self.residual_block2(out)  # N x 128 x 400 x 400
        out = self.residual_block3(out)  # N x 256 x 200 x 200
        out = self.residual_block4(out)  # N x 256 x 200 x 200
        out = self.residual_block5(out)  # N x 256 x 200 x 200
        out = self.avgpool(out)  # N x 256 x 1 x 1
        out = torch.flatten(out, 1)  # N x 256
        out = self.fc(out)  # N x 768
        out = torch.unsqueeze(out, 1)  # N x 1 x 768
        return out


class MultiViewLidarProcessor(nn.Module):

    def __init__(self, gpa_hidden_size, hidden_size, lm):
        super(MultiViewLidarProcessor, self).__init__()

        # Make MultiViewProcessor
        self.mvp = MultiViewProcessor(gpa_hidden_size, hidden_size, lm, lidar=True)

        # To extract features from  pseudo image
        self.pseudo_image_model = PseudoImageCNN()

        cfg = Config.fromfile('../mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py')
        self.lidar_model = init_model(cfg,
                                      '../mmdetection3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth')

        for param in self.lidar_model.parameters():
            param.requires_grad = False

        self.lidar_projection_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(0.1)
        )

    def get_lidar_embeddings(self, voxel_dict, batch_input_metas):
        # Get voxel features from encoder (N x 64)
        voxel_features = self.lidar_model.pts_voxel_encoder(voxel_dict['voxels'],
                                                            voxel_dict['num_points'],
                                                            voxel_dict['coors'], None,
                                                            batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1

        # Create pseudo image from voxel features (N x 64 x 400 x 400)
        lidar_features = self.lidar_model.pts_middle_encoder(voxel_features, voxel_dict['coors'], batch_size)

        # Extract features from pseudo image (N x 1 x 768)
        lidar_features = self.lidar_projection_layer(self.pseudo_image_model(lidar_features))

        # Add modal embeddings to lidar
        lidar_features += self.mvp.modal_embeddings(2 * torch.ones((1, lidar_features.shape[1]),
                                                                   dtype=torch.int, device=device))

        return lidar_features

    def forward(self, text_enc, imgs, voxel_dict, batch_input_metas, text_model):
        lidar_features = self.get_lidar_embeddings(voxel_dict, batch_input_metas)

        # Get combined text and image features using MVP (N x S x 768)
        merged_embedding = self.mvp(text_enc, imgs, text_model)

        # Combine lidar and text/image features (N x S + 1 x 768)
        merged_embedding = torch.cat([lidar_features, merged_embedding], dim=1)

        return merged_embedding
