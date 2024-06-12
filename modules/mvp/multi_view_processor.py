from torchvision.models import vit_b_32
import torch.nn as nn
import torch
from transformers import CLIPVisionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIT_HIDDEN_STATE = 768
CLIP_HIDDEN_STATE = 768
VIT_SEQ_LENGTH = 49
CLIP_SEQ_LENGTH = 50

class MultiViewProcessor(nn.Module):

    def __init__(self, gpa_hidden_size, hidden_size, lm, lidar=False):

        super().__init__()

        # Use ViT for image embeddings
        self.img_model = vit_b_32(weights='DEFAULT')
        self.lm = lm

        # Modal embedding to distinguish between image and text and lidar
        if lidar:
            self.modal_embeddings = nn.Embedding(3, hidden_size)
        else:
            self.modal_embeddings = nn.Embedding(2, hidden_size)

        self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        # If we are freezing the CLIP embeddings
        for param in self.img_model.parameters():
            param.requires_grad = False

        # Set matrices based on MIVC paper
        self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
        self.Z = nn.Sequential(
            nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
            nn.Tanh()
        )
        self.G = nn.Sequential(
            nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
            nn.Sigmoid()
        )

        self.img_projection_layer = nn.Sequential(
            nn.Linear(in_features=VIT_HIDDEN_STATE, out_features=2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.Dropout(0.1)
        )

    def gpa(self, img_embeddings):

        """"
        Calculates the gated-pooling attention score for the image embeddings
        :param img_embeddings: (6x768) dimensional
        :return single embedding of size (768,)
        """

        # Get weights for gated pooling attention
        gpa_weights = torch.softmax(self.w(self.Z(img_embeddings) * self.G(img_embeddings)), dim=0)

        # Take a linear combination of all the image embeddings
        fused_embeddings = torch.sum(gpa_weights * img_embeddings, dim=0)

        return fused_embeddings

    def get_img_embedding(self, imgs):

        N = imgs.shape[0]

        # Process into patches (N x 6 x 49 x H)
        merged_embedding = torch.stack([self.img_model._process_input(img) for img in imgs], dim=0)

        # Concatenate the batch class tokens -> (N, 6, 50, H)
        batch_class_tokens = self.img_model.class_token.expand(merged_embedding.shape[1], -1, -1).repeat(N, 1, 1, 1)
        merged_embedding = torch.cat([batch_class_tokens, merged_embedding], dim=2)

        # Add positional embeddings and remove class token -> (N, 6, 49, H)
        merged_embedding += self.img_model.encoder.pos_embedding.repeat(N, 1, 1, 1)
        merged_embedding = merged_embedding[:, :, 1:]

        # Get merged embedding and reshape to 2D embedding -> (N, 1, 49, H)
        merged_embedding = torch.stack([self.gpa(embedding.flatten(start_dim=1)).reshape(VIT_SEQ_LENGTH,
                                                                                         VIT_HIDDEN_STATE) for
                                        embedding in merged_embedding], dim=0)

        # Project to VL dimension -> (1, 49, H) (H is 512 for t5-small, 768 for t5-base)
        merged_embedding = self.img_projection_layer(merged_embedding)

        # Add modal type embedding to merged embedding
        merged_embedding += self.modal_embeddings(
            torch.ones((1, merged_embedding.shape[1]), dtype=torch.int, device=device))

        return merged_embedding

    def forward(self, text_enc, imgs, text_model):

        # Get the image embeddings (N x 1 x 49 x H)
        imgs_embedding = self.get_img_embedding(imgs)

        # Get the text embeddings (N x S x H)
        text_embeddings = text_model.get_input_embeddings()(text_enc)

        # Add modal embeddings to text
        text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                             device=device))

        # Concatenate embeddings -> (1 x S x 512)
        merged_embedding = torch.cat([imgs_embedding, text_embeddings], dim=1)

        return merged_embedding


class MultiViewProcessorCLIP(nn.Module):

    def __init__(self, gpa_hidden_size, hidden_size, lm, lidar=False):

        super().__init__()

        # Use ViT for image embeddings
        self.img_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_hidden_state = self.img_model.config.hidden_size
        self.lm = lm

        # Modal embedding to distinguish between image and text
        if lidar:
            self.modal_embeddings = nn.Embedding(3, hidden_size)
        else:
            self.modal_embeddings = nn.Embedding(2, hidden_size)

        self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.hidden_size = hidden_size

        # If we are freezing the CLIP embeddings
        for param in self.img_model.parameters():
            param.requires_grad = False

        # Set matrices based on MIVC paper
        self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
        self.Z = nn.Sequential(
            nn.Linear(in_features=self.clip_hidden_state * CLIP_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
            nn.Tanh()
        )
        self.G = nn.Sequential(
            nn.Linear(in_features=self.clip_hidden_state * CLIP_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
            nn.Sigmoid()
        )

        self.img_projection_layer = nn.Sequential(
            nn.Linear(in_features=self.clip_hidden_state, out_features=2 * self.clip_hidden_state),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2 * self.clip_hidden_state, hidden_size)
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

    def get_img_embedding(self, imgs):

        N = imgs.shape[0]

        # Get the hidden state CLIP output (N, 6, S, H)
        merged_embedding = torch.stack([self.img_model(pixel_values=img_batch).last_hidden_state for img_batch in imgs], dim=0)

        # Get merged embedding and reshape to 2D embedding -> (N, S, H)
        merged_embedding = torch.stack([self.gpa(embedding.flatten(start_dim=1)).reshape(
            CLIP_SEQ_LENGTH, self.clip_hidden_state) for embedding in merged_embedding], dim=0)

        # Project to LLM embedding dimension -> (N, S, 768)
        merged_embedding = self.img_projection_layer(merged_embedding)

        # Add modal type embedding to merged embedding
        merged_embedding += self.modal_embeddings(
            torch.ones((1, merged_embedding.shape[1]), dtype=torch.int, device=device))

        return merged_embedding

    def forward(self, text_enc, imgs, text_model):

        # Get the image embeddings (N x 50 x H)
        imgs_embedding = self.get_img_embedding(imgs)

        # Get the text embeddings (N x S x H)
        text_embeddings = text_model.get_input_embeddings()(text_enc)

        # Add modal embeddings to text
        text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                             device=device))

        # Concatenate embeddings -> (1 x S x 768)
        merged_embedding = torch.cat([imgs_embedding, text_embeddings], dim=1)

        return merged_embedding