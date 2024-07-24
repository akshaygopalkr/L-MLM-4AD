from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
from PIL import Image
from transformers import AutoImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiFrameDataset(Dataset):

    def __init__(self, config, input_file, tokenizer, challenge=False):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.challenge = challenge
        self.transform = []
        self.config = config

        if config.vit_patch:
            self.transform.append(AutoImageProcessor.from_pretrained(
                'google/vit-base-patch32-224-in21k', use_fast=True))
        if config.clip_patch or config.owl_patch:
            self.transform.append(AutoImageProcessor.from_pretrained('openai/clip-vit-base-patch32'))

        self.clip_and_owl = config.clip_patch and config.owl_patch

        if not self.transform:
            raise ValueError("At least one patch model (vit, clip, owl-vit) must be selected")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        qa, img_path = self.data[idx]

        img_path = [img_path[cam] for cam in sorted(list(img_path.keys()))]
        img_path = [os.path.join('data', 'nuscenes_all', '/'.join(p.split('/')[-3:])) for p in img_path]

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor -> (N_transforms, 6, 3, 224, 224)
        imgs = torch.stack([transform(images=[Image.open(p) for p in img_path], return_tensors='pt').pixel_values.to(device)
                             for transform in self.transform], dim=0)

        if self.clip_and_owl:
            imgs = torch.concat([imgs, imgs[-1].unsqueeze(0)], dim=0)

        if not self.challenge:
            return q_text, imgs, a_text, img_path
        else:
            return q_text, imgs, a_text, img_path, qa['id']

    def collate_fn(self, batch):

        q_texts, imgs, a_texts, _ = zip(*batch)

        # (N, N_transforms, 6, 3, 224, 224)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, imgs, labels

    def test_collate_fn(self, batch):

        if not self.challenge:
            q_texts, imgs, a_texts, img_path = zip(*batch)
        else:
            q_texts, imgs, a_texts, img_path, ids = zip(*batch)

        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        if not self.challenge:
            return list(q_texts), encodings, imgs, labels, img_path
        else:
            return list(q_texts), encodings, imgs, labels, img_path, list(ids)
