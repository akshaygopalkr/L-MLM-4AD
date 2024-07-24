import os
import pandas as pd

if __name__ == '__main__':

    patch_dict = {
        'CLIP-Patch': '--clip-patch',
        'OWL-Patch': '--owl-patch',
        'ViT-Patch': '--vit-patch'
    }

    for model_dir in os.listdir('multi_frame_results')[-4:]:

        if not os.path.exists(f'multi_frame_results/{model_dir}/multi_frame_results.csv'):
            patch_str = '--clip-patch --owl-patch --vit-patch'
        else:
            df = pd.read_csv(f'multi_frame_results/{model_dir}/multi_frame_results.csv')
            patch_list = [
                patch_dict[patch] for patch in list(patch_dict.keys()) if df[patch][0]
            ]
            patch_str = ' '.join(patch_list)

        os.system(
            f'python train.py --batch-size 8 --lora --epochs 10 {patch_str} --load-checkpoint --checkpoint-file {model_dir}'
        )

        os.system(
            f'python eval.py --batch-size 8 --lora  {patch_str} --model-name {model_dir}'
        )