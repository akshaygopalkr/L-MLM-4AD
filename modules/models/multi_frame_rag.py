import clip
import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
from PIL import Image
from modules.models.multi_frame_model import print_trainable_parameters
import numpy as np
from modules.models.multi_frame_model import MultiViewProcessorCLIP, MultiViewProcessor

CLIP_HIDDEN_STATE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MFRAG():

    def __init__(self, config, tokenizer):

        super().__init__()

        # Whether to use contrasting examples or not
        self.rag_contrast = config.rag_contrast
        self.num_context = config.num_context

        self.embedder, self.preprocess = clip.load('ViT-B/32', device=device)

        # Freeze the embedder
        for param in self.embedder.parameters():
            param.requires_grad = False

        # Load dataset to search based on embedding index
        self.embedding_dataset = load_dataset('agopalkr/DriveLM-RAG', split='train')
        self.embedding_dataset.add_faiss_index(column='embedding')

        self.tokenizer = tokenizer

    def get_rag_example(self, embeddings, mvp, text_model):

        # Stores the closest text
        closest_text = [[] for i in range(len(embeddings))]
        imgs = [[] for i in range(len(embeddings))]

        idx = 0

        for embedding in embeddings:

            scores, samples = self.embedding_dataset.get_nearest_examples(
                'embedding', embedding, k=100
            )

            # closest_answer = samples['text'][0].split('Answer:')[1]
            # closest_idx_2 = 1

            # if self.rag_contrast:
            #
            #     # Find the closest embedding that has a different response to the text
            #     for i in range(1, 100):
            #         if samples['text'][i].split('Answer:')[1] != closest_answer:
            #             closest_idx_2 = i
            #             break

            for context_idx in range(self.num_context):
                imgs[idx].append(torch.stack([self.preprocess(Image.open(p)).to(device) for p in samples['image_paths']
                                [context_idx]], dim=0))
                closest_text[idx].append(samples['text'][context_idx])


            idx += 1

        img_context = [torch.stack([imgs[i][idx] for i in range(len(imgs))]) for idx in range(self.num_context)]
        text_context = [self.tokenizer([closest_text[i][idx] for i in range(len(closest_text))], padding=True,
                                       return_tensors='pt').input_ids.to(device) for idx in range(self.num_context)]

        embedding_context = [mvp(text_context[i], img_context[i], text_model) for i in range(self.num_context)]
        rag_embedding = torch.concat(embedding_context, dim=1)

        return rag_embedding

    def get_embedding(self, text_batch, img_paths):
        """
        Makes the embedding used to search in the custom dataset
        text_batch: list of prompt text for the batch
        img_paths:  2D list of image paths for each view
        """

        with torch.no_grad():
            img_batch = torch.stack([torch.stack([self.preprocess(Image.open(view)).to(device)
                                                  for view in img_views], dim=0) for img_views in img_paths], dim=0)

            # Creates N x H embeddings of image and text
            image_features = torch.stack([torch.mean(self.embedder.encode_image(imgs), dim=0) for imgs in img_batch])
            text_enc = clip.tokenize(text_batch, truncate=True).to(device)
            text_features = self.embedder.encode_text(text_enc)

            # Concatenate text and image embeddings into
            embedding = torch.mean(torch.stack([image_features, text_features], dim=1),
                                   dim=1).detach().cpu().numpy().astype(np.float32)

        return embedding


class DriveVLMT5(nn.Module):

    def __init__(self, config, tokenizer):

        super().__init__()

        model_str = 'google-t5/t5-base' if config.lm == 'T5-Base' else 'google-t5/t5-large'
        self.model = T5ForConditionalGeneration.from_pretrained(model_str)

        # model_str = 'EleutherAI/pile-t5-base' if config.lm == 'T5-Base' else 'EleutherAI/pile-t5-large'
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_str)

        hidden_size = self.model.config.d_model

        # If we are freezing the CLIP embeddings
        if config.freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False

        self.rag = config.rag

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        if config.img_encoder == 'CLIP':

            # Create instance for multi-view processor
            self.mvp = MultiViewProcessorCLIP(config.gpa_hidden_size, hidden_size, config.lm)

        else:
            self.mvp = MultiViewProcessor(config.gpa_hidden_size, hidden_size, config.lm)

        if self.rag:
            # For RAG Model
            self.rag_model = MFRAG(config, tokenizer)

    def forward(self, text_enc, imgs, text, img_paths, labels=None):

        merged_embedding = self.mvp(text_enc, imgs, self.model)

        if self.rag:

            # Get the rag embeddings
            dataset_embeddings = self.rag_model.get_embedding(text, img_paths)
            rag_embeddings = self.rag_model.get_rag_example(dataset_embeddings, self.mvp, self.model)

            # Get the merged embeddings and concatenate them with the RAG embeddings
            merged_embedding = torch.concat([rag_embeddings, merged_embedding], dim=1)

        # If training include the labels
        return self.model(inputs_embeds=merged_embedding, labels=labels)
