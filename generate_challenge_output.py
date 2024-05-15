import argparse

import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from modules.dataset.multi_frame_dataset import MultiFrameDataset
from modules.dataset.multi_frame_lidar_dataset import MultiFrameLidarDataset
from modules.dataset.multi_video_lidar_dataset import MultiVideoLidarDataset
from modules.dataset.multi_video_dataset import MultiVideoDataset
from modules.models.multi_frame_model import DriveVLMT5 as ImageModel
from modules.models.multi_video_model import DriveVLMT5 as VideoModel
from modules.config.eval_config import params
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
import json
import pandas as pd
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val_model(dloader):
    model.eval()
    ids_answered = set()
    test_data = []

    with torch.no_grad():
        for idx, (batch) in progress_bar(enumerate(dloader), total=len(dloader)):

            # Forward pass through model
            if not config.lidar:
                q_texts, inputs, imgs, labels, img_paths, ids = batch
                outputs = model.generate(inputs, imgs)
            else:
                inputs, imgs, lidar, labels, img_paths, ids = batch
                outputs = model.generate(inputs, imgs, lidar)

            # Get the text output
            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

            if idx % 100 == 0:
                print(q_texts)
                print(text_outputs)

            for q_text, text_output, q_id in zip(q_texts, text_outputs, ids):

                test_data.append(
                    {
                        'id': q_id,
                        'question': q_text,
                        'gt_answer': '',
                        'answer': text_output
                    }
                )

    # Save test output to file
    with open(os.path.join('multi_frame_results', config.model_name, pred_fname), 'w') as f:
        json.dump(test_data, f)


if __name__ == '__main__':

    config = params()

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')

    if config.video:

        model_type = VideoModel
        if config.lidar:
            data_folder = 'multi_video_LIDAR'
            dataset = MultiVideoLidarDataset
            test_file = f'multi_video_LIDAR_test_challenge_only_q.json'
        else:
            data_folder = 'multi_video'
            dataset = MultiVideoDataset
            test_file = f'multi_video_test_challenge_only_q.json'

    # Make datasets and model for image + lidar
    else:

        model_type = ImageModel

        if config.lidar:
            dataset = MultiFrameLidarDataset
            data_folder = 'multi_frame_LIDAR'
            test_file = f'multi_frame_LIDAR_test_challenge_only_q.json'
        else:
            dataset = MultiFrameDataset
            data_folder = 'multi_frame'
            test_file = f'multi_frame_test_challenge_only_q.json'

    model = model_type(config)

    model.load_state_dict(
        torch.load(os.path.join('multi_frame_results', config.model_name,
                                'latest_model.pth')))

    # model = torch.load(os.path.join('multi_frame_results', config.model_name, 'latest_model.pth'))
    model.to(device)

    # Load dataset and dataloader
    test_dset = dataset(
        input_file=os.path.join('data', data_folder, test_file),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ]),
        challenge=True
    )
    test_dloader = DataLoader(test_dset, shuffle=True, batch_size=config.batch_size, collate_fn=test_dset.test_collate_fn)

    # Load in image ids
    with open(os.path.join('data', 'multi_frame', f'image_id_{config.dataset}.json')) as f:
        image_id_dict = json.load(f)

    pred_fname = 'output.json'

    # Get the loss and predictions from the model
    val_model(test_dloader)
