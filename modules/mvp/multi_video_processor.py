import pdb

from transformers import XCLIPVisionModel
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_STATE = 768
SEQ_LENGTH = 49


class MultiVideoProcessor(nn.Module):

    def __init__(self, gpa_hidden_size, hidden_size, lm, lidar=False):

        super().__init__()

        # Use ViT for image embeddings
        self.patch_embedding = nn.Conv2d(in_channels=24, out_channels=768, kernel_size=32, stride=32)
        self.lm = lm

        # Modal embedding to distinguish between image and text
        if not lidar:
            self.modal_embeddings = nn.Embedding(2, hidden_size)
        else:
            self.modal_embeddings = nn.Embedding(3, hidden_size)

        self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden_size = hidden_size

        # Set matrices based on MIVC paper
        self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
        self.Z = nn.Sequential(
            nn.Linear(in_features=SEQ_LENGTH*HIDDEN_STATE, out_features=gpa_hidden_size, bias=False),
            nn.Tanh()
        )
        self.G = nn.Sequential(
            nn.Linear(in_features=SEQ_LENGTH*HIDDEN_STATE, out_features=gpa_hidden_size, bias=False),
            nn.Sigmoid()
        )

        self.vid_projection_layer = nn.Sequential(
            nn.Linear(in_features=HIDDEN_STATE, out_features=2*self.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size)
        )

    def gpa(self, img_embeddings):

        """"
        Calculates the gated-pooling attention score for the image embeddings
        :param img_embeddings: (6x512) dimensional
        :return single embedding of size (512,)
        """

        # Get weights for gated pooling attention
        gpa_weights = torch.softmax(self.w(self.Z(img_embeddings) * self.G(img_embeddings)), dim=0)

        # Take a linear combination of all the image embeddings
        fused_embeddings = torch.sum(gpa_weights * img_embeddings, dim=0)

        return fused_embeddings

    def get_video_embedding(self, vids):

        # Images are of shape (N, 6, 3*8, 224, 224) -> output is (N, 6, S, 768)
        vid_embedding = torch.stack([self.patch_embedding(view)
                              for view in vids], dim=0).flatten(start_dim=3).permute(1, 0, 3, 2)

        # Get merged embedding and reshape to 2D embedding -> (N, S, 768)
        merged_embedding = torch.stack([self.gpa(embedding.flatten(start_dim=1)).reshape(
            SEQ_LENGTH, HIDDEN_STATE) for embedding in vid_embedding], dim=0)

        # Project to LLM embedding dimension -> (N, S, 768)
        merged_embedding = self.vid_projection_layer(merged_embedding)

        # Add modal type embedding to merged embedding
        merged_embedding += self.modal_embeddings(
            torch.ones((1, merged_embedding.shape[1]), dtype=torch.int, device=device))

        return merged_embedding

    def forward(self, text_enc, imgs, text_model):

        # Get the image embeddings (N x S_I x H)
        video_embedding = self.get_video_embedding(imgs)

        # Get the text embeddings (N x S_T x H)
        text_embeddings = text_model.get_input_embeddings()(text_enc)

        # Add modal embeddings to text
        text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                             device=device))

        # Concatenate embeddings -> (1 x (S_I + S_T) x H)
        merged_embedding = torch.cat([video_embedding, text_embeddings], dim=1)

        return merged_embedding