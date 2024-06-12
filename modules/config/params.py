import argparse
import os

def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Model learning rate starting point, default is 1e-4.")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--weight-decay", default=0.05, type=float,
                        help="L2 Regularization, default is 0.05")
    parser.add_argument("--epochs", default=12, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument("--hf-train", action='store_true',
                        help="Whether to use HuggingFace default training or custom training loop")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--checkpoint-frequency', default=10000, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--lora', action='store_true', help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--dataset', default='DriveLM-challenge', type=str, choices=['all', 'DriveLM', 'NuScenesQA', 'DriveLM-challenge'],
                        help='Can either perform training on all datasets or a specific dataset')
    parser.add_argument('--img-encoder', type=str, choices=['CLIP', 'Patch'], default='Patch', help='Image encoder '
                                                                                   'to be used for the network')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')

    # check if multi_frame_results is not empty
    if len(os.listdir('multi_frame_results')) == 0:
        most_recent_file = ''
    else:
        most_recent_file = sorted(os.listdir('multi_frame_results'))[-1]

    parser.add_argument('--load-checkpoint', action='store_true', help='Whether to load a checkpoint from '
                                                                       'multi_frame_results folder')
    parser.add_argument('--checkpoint-file', default=most_recent_file, type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')
    parser.add_argument('--video', action='store_true', help='Whether to use video data')
    parser.add_argument('--lidar', action='store_true', help='Whether to use LiDAR data')

    args = parser.parse_args()
    return args