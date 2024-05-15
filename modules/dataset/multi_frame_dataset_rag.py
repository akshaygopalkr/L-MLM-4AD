from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
import clip
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiFrameDataset(Dataset):

    def __init__(self, input_file, tokenizer):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        _ , self.preprocess = clip.load('ViT-B/32', device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        qa, img_path = self.data[idx]
        img_path = list(img_path.values())

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor
        imgs = torch.stack([self.preprocess(Image.open(img)).to(device) for img in img_path], dim=0)

        return q_text, imgs, a_text, sorted(list(img_path))

    def collate_fn(self, batch):
        q_texts, imgs, a_texts, img_paths = zip(*batch)
        q_texts, imgs, a_texts, _ = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, imgs, labels, list(q_texts), list(img_paths)
        # return encodings, imgs, labels

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return list(q_texts), encodings, imgs, labels, img_path
