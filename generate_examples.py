import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import argparse
import random
from PIL import Image
import numpy as np
import cv2
import textwrap
import matplotlib.image as mpimg

mpl.use('TkAgg')  # !IMPORTANT


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='T5-Base')
    parser.add_argument('--num-examples', type=int, default=750)

    args = parser.parse_args()
    return args


def show_prediction(images, question, answer, gt, idx):
    # Create a figure and axis
    fig, ax = plt.subplots(nrows=4, ncols=3, gridspec_kw={'hspace': 0, 'wspace': 0.05}, figsize=(12,9))
    # ax[1, 1].remove()
    # ax[2, 1].remove()
    plt.tight_layout()

    for i in range(4):
        for j in range(3):
            ax[i, j].axis('off')

    images = [images['CAM_FRONT_LEFT'], images['CAM_FRONT'], images['CAM_FRONT_RIGHT'], images['CAM_BACK_LEFT'], images['CAM_BACK'], images['CAM_BACK_RIGHT']]
    indices_arr = [(1, 0), (0, 1), (1, 2), (2, 0), (3, 1), (2, 2)]

    img_count = 0

    for i in range(2):
        for j in range(3):
            img = mpimg.imread(images[img_count])
            row, col = indices_arr[img_count]
            ax[row, col].imshow(img)
            img_count += 1

    # Plotting the large image
    ax_large = fig.add_subplot(4,3, (5,8))
    img_large = mpimg.imread('images.jpeg')
    ax_large.imshow(img_large)
    ax_large.axis('off')

    fig.subplots_adjust(hspace=0, wspace=0)

    wrap_width = 125
    question = textwrap.fill(question, wrap_width)

    # Add the question
    plt.figtext(0.5, 0.95, question, horizontalalignment='center', verticalalignment='center',
                fontsize=12, family='serif')

    answer = 'Prediction: ' + answer
    answer = textwrap.fill(answer, wrap_width)
    gt = 'Ground Truth: ' + gt
    gt = textwrap.fill(gt, wrap_width)

    # Add the answer
    plt.figtext(0.5, 0.05, answer, ha='center',
                fontsize=12, family='serif')
    plt.figtext(0.5, 0.02, gt, ha='center',
                fontsize=12, family='serif')

    plot_path = os.path.join('multi_frame_results', args.date, 'imgs', str(idx) + '.png')
    # plt.show()
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':

    args = params()

    # Load in image ids
    with open(os.path.join('data', 'multi_frame', 'image_id.json')) as f:
        image_id_dict = json.load(f)
    with open(os.path.join('data', 'multi_frame', 'multi_frame_test_coco.json')) as f:
        data = json.load(f)

    # Make a reverse dictionary to map to images
    image_id_to_img = {}

    for key in image_id_dict:
        # Get the image path
        image_id, img_paths = image_id_dict[key]
        img_path = key[:key.index('.jpg') + 4]
        question = key[key.index('.jpg') + 4:]
        image_id_to_img[image_id] = (img_paths, question)

    with open(os.path.join('multi_frame_results', args.date, 'predictions.json')) as f:
        predictions = json.load(f)

    if not os.path.exists(os.path.join('multi_frame_results', args.date, 'imgs')):
        os.mkdir(os.path.join('multi_frame_results', args.date, 'imgs'))

    # Shuffle the predictions
    random.shuffle(predictions)

    for i in range(args.num_examples):
        prediction = predictions[i]
        img_id, answer = prediction['image_id'], prediction['caption']
        gt = data['annotations'][img_id]['caption']
        img_files, question = image_id_to_img[img_id]
        question = f'Image Id: {img_id} {question}'
        show_prediction(img_files, question, answer, gt, i)
