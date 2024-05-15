from torch.utils.data import Dataset
from transformers import AutoImageProcessor
import json
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiVideoDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.preprocess = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        # img_paths: list of lists for camera views
        qa, img_paths = self.data[idx]

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor
        videos = torch.stack([torch.stack([self.preprocess(Image.open(img)).to(device) for img in cam], dim=0)
                              for cam in img_paths], dim=0)

        return q_text, videos, a_text, img_paths

    def collate_fn(self, batch):

        q_texts, videos, a_texts, _ = zip(*batch)
        videos = torch.stack(list(videos), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return encodings, videos, labels

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        return list(q_texts), encodings, imgs, labels, img_path