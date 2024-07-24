from torch.utils.data import Dataset
from transformers import AutoProcessor
from torchvision.io import read_image
import json
import os
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiVideoDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.preprocess = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        # img_paths: list of lists for camera views
        qa, img_paths = self.data[idx]

        img_paths = {cam: [os.path.join('/home/cvrr/Desktop/docker/BevFusion_AL', '/'.join(img_path.split('/')[3:]))
                           for img_path in img_paths[cam]] for cam in img_paths}

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor (6 x (V*3) x 224 x 224)
        videos = torch.stack([torch.concat([self.transform(read_image(img).float().to(device)) for img in img_paths[cam]])
                              for cam in img_paths], dim=0)

        return q_text, videos, a_text

    def collate_fn(self, batch):

        q_texts, videos, a_texts = zip(*batch)

        # Stack the videos into a single tensor (N x 6 x (V x 3) x 224 x 224)
        videos = torch.stack(list(videos), dim=0)

        # Reshape to (6 x N x (V x 3) x 224 x 224)
        videos = videos.transpose(0, 1)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, videos, labels

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return list(q_texts), encodings, imgs, labels, img_path