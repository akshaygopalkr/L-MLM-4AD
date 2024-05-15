from torch.utils.data import Dataset
from transformers import AutoImageProcessor
import json, torch
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmengine.config import Config
from mmdet3d.apis import init_model
from copy import deepcopy
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiVideoLidarDataset(Dataset):

    def __init__(self, input_file, tokenizer, transform=None):
        with open(input_file) as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.preprocess = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        # Get the question and answer at the idx
        # img_paths: list of lists for camera views
        qa, data_paths = self.data[item]
        img_paths = [data_paths[cam] for cam in self.cam_names]

        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"

        # Concatenate images into a single tensor
        videos = torch.stack([torch.stack([self.preprocess(Image.open(img)).to(device) for img in cam], dim=0)
                              for cam in img_paths], dim=0)

        # Get the paths to the LiDAR data
        lidar_paths = data_paths['LIDAR_TOP']

        # List of length T
        lidar_data = [torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(1, -1)) for lidar_path in
                      lidar_paths]

        return q_text, videos, a_text, lidar_data, data_paths

    def collate_fn(self, batch):

        q_texts, videos, a_texts, lidar_data, _ = zip(*batch)
        videos = torch.stack(list(videos), dim=0)

        encodings = self.tokenizer(q_texts, padding=True, return_tensors="pt").input_ids.to(device)
        labels = self.tokenizer(a_texts, padding=True, return_tensors='pt').input_ids.to(device)

        # Conver to T x N x H list
        lidar_data_list = [[lidar_data[j][i] for j in range(len(lidar_data[i]))] for i in range(len(lidar_data))]

        voxel_dict_list, batch_input_metas_list = [], []

        for lidar_data in lidar_data_list:

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
            voxel_dict_list.append(voxel_dict), batch_input_metas_list.append(batch_input_metas)

        return encodings, videos, (voxel_dict_list, batch_input_metas_list), labels

    def test_collate_fn(self, batch):

        encodings, videos, (voxel_dict_list, batch_input_metas_list), labels = self.collate_fn(batch)
        q_texts, _, _, _, data_paths = zip(*batch)

        return list(q_texts), encodings, videos, (voxel_dict_list, batch_input_metas_list), labels, data_paths
