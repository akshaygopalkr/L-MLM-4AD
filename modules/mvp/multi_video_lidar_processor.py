import torch.nn as nn
from transformers import TimesformerModel
from modules.mvp.multi_video_processor import MultiVideoProcessor
from modules.mvp.multi_view_lidar_processor import PseudoImageCNN
from mmengine.config import Config
from mmdet3d.apis import init_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LIDAR_HIDDEN_STATE = 768


class MultiVideoLidarProcessor(nn.Module):

    def __init__(self, gpa_hidden_size, hidden_size, lm):
        super(MultiVideoLidarProcessor, self).__init__()

        self.mvp = MultiVideoProcessor(gpa_hidden_size, hidden_size, lm, lidar=True)

        # Modal embedding to distinguish between image, lidar and text
        self.modal_embeddings = nn.Embedding(3, hidden_size)

        self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden_size = hidden_size

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

        # Set matrices based on MIVC paper
        self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
        self.Z = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=gpa_hidden_size, bias=False),
            nn.Tanh()
        )
        self.G = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=gpa_hidden_size, bias=False),
            nn.Sigmoid()
        )

    def lidar_gpa(self, lidar_embeddings):
        """"
        Calculates the gated-pooling attention score for the lidar embeddings
        :param lidar_embeddings: (Tx768) dimensional
        :return single embedding of size (768,)
        """

        # Get weights for gated pooling attention
        gpa_weights = torch.softmax(self.w(self.Z(lidar_embeddings) * self.G(lidar_embeddings)), dim=0)

        # Take a linear combination of all the lidar embeddings
        fused_embeddings = torch.sum(gpa_weights * lidar_embeddings, dim=0)

        return fused_embeddings

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

    def forward(self, text_enc, imgs, voxel_dicts, batch_input_metas, text_model):
        # Get the lidar features (N x T x 768)
        lidar_features = torch.stack([self.get_lidar_embeddings(voxel_dict, batch_input_meta)
                                      for voxel_dict, batch_input_meta in zip(voxel_dicts, batch_input_metas)], dim=1)

        lidar_features = torch.stack([self.lidar_gpa(embedding) for embedding in lidar_features], dim=0)

        # Get combined text and image features using MVP (N x S x 768)
        merged_embedding = self.mvp(text_enc, imgs, text_model)

        # Combine lidar and text/image features (N x S + 1 x 768)
        merged_embedding = torch.cat([lidar_features, merged_embedding], dim=1)

        return merged_embedding
