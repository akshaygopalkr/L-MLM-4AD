import pdb

from nuscenes.nuscenes import NuScenes
import os
import pdb
import json


def make_video_data(data, lidar=False):
    new_data = []
    datasets = [nusc_unlabeled, nusc, nusc_test]

    data_copy = data.copy()

    for item in data_copy:

        dataset = None
        sample_token = path_to_token[item[1]['CAM_FRONT'].split('/')[-1]]

        for i in range(len(datasets)):
            try:
                sample_data = datasets[i].get('sample', sample_token)
                dataset = datasets[i]
                break
            except KeyError:
                pass

        cam_dict = {cam: [] for cam in cam_names}

        for cam in cam_names:

            camera_data = dataset.get('sample_data', sample_data['data'][cam])
            img_filepath = os.path.join('/mnt/datasets/data/nuscenes', camera_data['filename'])
            cam_dict[cam].append(img_filepath)

            # Get previous camera frames
            prev_camera_data = camera_data.copy()
            for i in range(7):
                prev_camera_data = dataset.get('sample_data', prev_camera_data['prev'])
                img_filepath = os.path.join('/mnt/datasets/data/nuscenes', prev_camera_data['filename'])
                cam_dict[cam].insert(0, img_filepath)

            # next_camera_data = camera_data.copy()
            #
            # for i in range(4):
            #     next_camera_data = dataset.get('sample_data', next_camera_data['next'])
            #     img_filepath = os.path.join('/mnt/datasets/data/nuscenes', next_camera_data['filename'])
            #     cam_dict[cam].append(img_filepath)

        if lidar:

            cam_dict['LIDAR_TOP'] = []
            lidar_data = dataset.get('sample_data', sample_data['data']['LIDAR_TOP'])
            cam_dict['LIDAR_TOP'].append(os.path.join('/mnt/datasets/data/nuscenes', lidar_data['filename']))

            prev_lidar_data = lidar_data.copy()
            for i in range(7):
                prev_lidar_data = dataset.get('sample_data', prev_lidar_data['prev'])
                cam_dict['LIDAR_TOP'].insert(0,
                                             os.path.join('/mnt/datasets/data/nuscenes', prev_lidar_data['filename']))

            # next_lidar_data = lidar_data.copy()
            # for i in range(4):
            #     next_lidar_data = dataset.get('sample_data', next_lidar_data['next'])
            #     cam_dict['LIDAR_TOP'].append(os.path.join('/mnt/datasets/data/nuscenes', next_lidar_data['filename']))

        new_data.append((item[0], cam_dict))

    return new_data


def make_mf_data(data, lidar=True):
    new_data = []
    datasets = [nusc_unlabeled, nusc, nusc_test]

    data_copy = data.copy()

    for item in data_copy:

        dataset = None
        sample_token = path_to_token[item[1]['CAM_FRONT'].split('/')[-1]]

        for i in range(len(datasets)):
            try:
                sample_data = datasets[i].get('sample', sample_token)
                dataset = datasets[i]
                break
            except KeyError as e:
                pass

        cam_dict = item[1]

        if lidar:
            lidar_data = dataset.get('sample_data', sample_data['data']['LIDAR_TOP'])
            cam_dict['LIDAR_TOP'] = os.path.join('/mnt/datasets/data/nuscenes', lidar_data['filename'])

        new_data.append((item, cam_dict))

    return new_data

if __name__ == '__main__':

    nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/datasets/data/nuscenes', verbose=True)
    nusc_test = NuScenes(version='v1.0-test', dataroot='/mnt/datasets/data/nuscenes', verbose=True)
    nusc_unlabeled = NuScenes(version='v1.0-unlabeled', dataroot='/mnt/datasets/data/nuscenes', verbose=True)

    path_to_token = {data['filename'].split('/')[-1]: data['sample_token'] for data in nusc.sample_data if
                     'samples' in data['filename']}
    path_to_token.update({data['filename'].split('/')[-1]: data['sample_token'] for data in nusc_test.sample_data if
                          'samples' in data['filename']})
    path_to_token.update(
        {data['filename'].split('/')[-1]: data['sample_token'] for data in nusc_unlabeled.sample_data if
         'samples' in data['filename']})

    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    with open(os.path.join('data', 'multi_frame', 'multi_frame_train_DriveLM-challenge.json')) as f:
        train_data = json.load(f)
    with open(os.path.join('data', 'multi_frame', 'multi_frame_test_challenge_only_q.json')) as f:
        test_data = json.load(f)
    with open(os.path.join('data', 'multi_frame', 'multi_frame_val_DriveLM-challenge.json')) as f:
        val_data = json.load(f)

    # drivelm_lidar_train = make_mf_data(train_data, True)
    # drivelm_lidar_val = make_mf_data(val_data, True)
    # drivelm_lidar_test = make_mf_data(test_data, True)

    drivelm_video_train = make_video_data(train_data)
    drivelm_video_val = make_video_data(val_data)
    # drivelm_video_test = make_video_data(test_data)

    drivelm_video_lidar_train = make_video_data(train_data, True)
    drivelm_video_lidar_test = make_video_data(test_data, True)
    # drivelm_video_lidar_val = make_video_data(val_data, True)

    # with open(os.path.join('data', 'multi_frame_LIDAR', 'multi_frame_LIDAR_train_DriveLM-challenge.json'), 'w') as f:
    #     json.dump(drivelm_lidar_train, f, indent=6)
    # with open(os.path.join('data', 'multi_frame_LIDAR', 'multi_frame_LIDAR_val_DriveLM-challenge.json'), 'w') as f:
    #     json.dump(drivelm_lidar_val, f, indent=6)
    # with open(os.path.join('data', 'multi_frame_LIDAR', 'multi_frame_test_challenge_only_q.json'), 'w') as f:
    #     json.dump(drivelm_lidar_test, f)

    with open(os.path.join('data', 'multi_video', 'multi_video_train_DriveLM-challenge.json'), 'w') as f:
        json.dump(drivelm_video_train, f, indent=6)
    with open(os.path.join('data', 'multi_video', 'multi_video_val_DriveLM-challenge.json'), 'w') as f:
        json.dump(drivelm_video_val, f, indent=6)
    # with open(os.path.join('data', 'multi_video', 'multi_video_test_challenge_only_q.json'), 'w') as f:
    #     json.dump(drivelm_video_test, f)

    # with open(os.path.join('data', 'multi_video_LIDAR', 'multi_video_LIDAR_train_DriveLM-challenge.json'), 'w') as f:
    #     json.dump(drivelm_video_lidar_train, f, indent=6)
    # with open(os.path.join('data', 'multi_video_LIDAR', 'multi_video_LIDAR_val_DriveLM-challenge.json'), 'w') as f:
    #     json.dump(drivelm_video_lidar_val, f, indent=6)
    # with open(os.path.join('data', 'multi_video_LIDAR', 'multi_video_LIDAR_test_challenge_only_q.json'), 'w') as f:
    #     json.dump(drivelm_video_lidar_test, f)