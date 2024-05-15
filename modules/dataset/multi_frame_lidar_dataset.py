import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch
import mmdet3d
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmengine.config import Config
from mmdet3d.apis import init_model
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiFrameLidarDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None, challenge=False):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.transform = transform
        self.cam_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                          'CAM_BACK', 'CAM_BACK_RIGHT']

        cfg = Config.fromfile('../mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py')
        model = init_model(cfg,
                           '../mmdetection3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth')
        self.lidar_preprocessor = model.data_preprocessor

        cfg.test_dataloader.dataset.pipeline[0].type = 'LoadPointsFromDict'

        # build the data pipeline
        test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
        self.test_pipeline = Compose(test_pipeline)
        self.box_type_3d, self.box_mode_3d = \
            get_box_type(cfg.test_dataloader.dataset.box_type_3d)
        self.challenge = challenge

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the question and answer at the idx
        qa, data_path = self.data[idx][0]
        img_path = [data_path[cam] for cam in self.cam_names]
        lidar_path = data_path['LIDAR_TOP']
        lidar_data = torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(1, -1))

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor
        imgs = [self.transform(read_image(p).float()).to(device) for p in img_path]
        imgs = torch.stack(imgs, dim=0)

        if not self.challenge:
            return q_text, imgs, a_text, lidar_data, sorted(list(img_path))
        else:
            return q_text, imgs, a_text, lidar_data, sorted(list(img_path)), qa['id']

    def collate_fn(self, batch):

        if not self.challenge:
            q_texts, imgs, a_texts, lidar_data, _ = zip(*batch)
        else:
            q_texts, imgs, a_texts, lidar_data, _, ids = zip(*batch)

        imgs = torch.stack(list(imgs), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        lidar_data = list(lidar_data)

        data = []

        for pcd in lidar_data:
            # directly use loaded point cloud
            data_ = dict(
                points=pcd,
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d)

            data_ = self.test_pipeline(data_)
            data.append(data_)

        collate_data = pseudo_collate(data)
        data = self.lidar_preprocessor(collate_data, False)

        batch_inputs_dict, batch_data_samples = data['inputs'], data['data_samples']
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        voxel_dict = batch_inputs_dict.get('voxels', None)

        if not self.challenge:
            return encodings, imgs, (voxel_dict, batch_input_metas), labels
        else:
            return encodings, imgs, (voxel_dict, batch_input_metas), labels, list(ids)

    def test_collate_fn(self, batch):

        if not self.challenge:
            encodings, imgs, (voxel_dict, batch_input_metas), labels = self.collate_fn(batch)
            q_texts, _, _, _, img_path = zip(*batch)
        else:
            encodings, imgs, (voxel_dict, batch_input_metas), labels, ids = self.collate_fn(batch)
            q_texts, _, _, _, img_path, _, _ = zip(*batch)

        return list(q_texts), encodings, imgs, (voxel_dict, batch_input_metas), labels, img_path
