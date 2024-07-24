from transformers import T5Tokenizer, TrainingArguments, Trainer
from transformers import AutoTokenizer
from modules.config.params import params
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch
from modules.dataset.multi_frame_dataset import MultiFrameDataset
from modules.dataset.multi_video_dataset import MultiVideoDataset
from modules.models.multi_video_model import DriveVLMT5 as VideoModel
from modules.models.multi_frame_model import print_trainable_parameters
from modules.models.multi_frame_model import DriveVLMT5 as ImageModel
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, model_name):
    # Save the model into the designated folder
    path = os.path.join('multi_frame_results', timestr, model_name + '.pth')
    torch.save(model, path)
    wandb.save(path)


def val_model(dloader, val_model):
    val_model.eval()
    val_loss = 0

    for idx, (batch) in tqdm(enumerate(dloader), total=len(dloader)):

        if not config.lidar:
            inputs, imgs, labels = batch
            outputs = val_model(inputs, imgs, labels)
        else:
            inputs, imgs, lidar, labels = batch
            outputs = val_model(inputs, imgs, labels, lidar)

        val_loss += outputs.loss.item()

    return val_loss / len(val_dataloader)


def save_stats(train_loss, val_loss, epochs, lr):
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_loss,
        'min val loss': val_loss,
        'epochs': epochs,
        'learning rate': lr,
        'LM': 'T5-Base',
        'Image Embedding': 'Patch'
    }

    # Save stats into checkpoint
    with open(os.path.join('multi_frame_results', timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def plot_loss(training_loss, val_loss):
    num_epochs = len(training_loss)

    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('multi_frame_results', timestr, 'loss.png'))


def custom_train(train_loss, val_loss, best_model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

    for epoch in range(epochs, config.epochs):
        print('-------------------- EPOCH ' + str(epoch) + ' ---------------------')
        model.train()
        epoch_loss = 0

        for step, (batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # Forward pass through model
            if not config.lidar:
                inputs, imgs, labels = batch
                outputs = model(inputs, imgs, labels)
            else:
                inputs, imgs, lidar, labels = batch
                outputs = model(inputs, imgs, labels, lidar)

            # Calculate loss
            loss = outputs.loss
            epoch_loss += loss.item()

            if step % config.checkpoint_frequency == 0:
                print()
                print('Loss: ' + str(loss.item()))

                # Get the hidden states (output)
                hidden_states = outputs.logits

                # Perform decoding (e.g., greedy decoding)
                outputs = torch.argmax(hidden_states, dim=-1)
                try:
                    text_outputs = [processor.decode(output.to('cpu'), skip_special_tokens=True) for output in outputs]
                    text_questions = [processor.decode(q.to('cpu'), skip_special_tokens=True) for q in inputs]
                    text_labels = [processor.decode(a.to('cpu'), skip_special_tokens=True) for a in labels]
                except IndexError as e:
                    print('Error decoding text')
                    print(text_outputs, text_questions, text_labels)

                wandb.log({'Training Loss': loss.item()})
                print()
                print('Questions:')
                print(text_questions)
                print()
                print('Generated Answers:')
                print(text_outputs)
                print()
                print('Ground Truth Answers:')
                print(text_labels)

            # Back-propogate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Get train and val loss per batch
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)

        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)

        wandb.log({'Validation Loss': epoch_val_loss})
        wandb.log({'Epoch Training Loss': epoch_train_loss})

        if not val_loss or min(epoch_val_loss, val_loss) == epoch_val_loss:
            val_loss = epoch_val_loss
            best_model = deepcopy(model.state_dict())
        if not train_loss or min(train_loss, epoch_train_loss) == epoch_train_loss:
            train_loss = epoch_train_loss

        # Adjust learning rate scheduler
        scheduler.step()

        print('Training Loss: ' + str(epoch_train_loss))
        print('Validation Loss: ' + str(epoch_val_loss))
        print('---------------------------------------------')

        # Save model and stats for checkpoints
        save_model(best_model, 'latest_model')
        epochs += 1
        save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0])

    # Save the model and plot the loss
    plot_loss(losses, val_losses)
    return train_loss, val_loss


def train():
    training_config = TrainingArguments(
        output_dir="agopalkr/EfficientDriveLM",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dset,
        eval_dataset=val_dset,
    )

    trainer.train()
    model.push_to_hub("agopalkr/EfficientDriveLM")


def save_experiment(statistics):
    """
    Saves the experiment multi_frame_results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA finetuning': [config.lora],
        'GPA Hidden Size': [config.gpa_hidden_size],
        'Video': [config.video],
        'LiDAR': [config.lidar],
        'CLIP-Patch': [config.clip_patch],
        'OWL-Patch': [config.owl_patch],
        'ViT-Patch': [config.vit_patch],
        'Freeze T5': [config.freeze_lm],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Min Testing Loss': [statistics[2]],
    }

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join('multi_frame_results', timestr, 'multi_frame_results.csv'), index=False, header=True)


if __name__ == '__main__':

    timestr = time.strftime("%Y%m%d-%H%M%S")

    config = params()

    patch_type = ''

    if config.vit_patch:
        patch_type += 'ViT-'
    if config.owl_patch:
        patch_type += 'OWL-'
    if config.clip_patch:
        patch_type += 'CLIP-'

    run_name = f'{patch_type}{timestr}'

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="L-MLM-4AD",
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "Model Name": timestr,
            "CLIP Patches": config.clip_patch,
            "ViT Patches": config.vit_patch,
            "OWL Patches": config.owl_patch,
            "LoRA": config.lora,
            "GPA Hidden Size": config.gpa_hidden_size,
            "Freeze LM": config.freeze_lm,
            "Video": config.video,
            "Learning Rate": config.learning_rate,
            "Weight decay": config.weight_decay,
            "Batch Size": config.batch_size,
            "epochs": config.epochs,
        }
    )

    if config.video:

        model_type = VideoModel
        if config.lidar:
            pass
        else:
            data_folder = 'multi_video'
            dataset = MultiVideoDataset
            train_file, val_file, test_file = (f'multi_video_train_{config.dataset}.json',
                                                  f'multi_video_val_{config.dataset}.json',
                                                  f'multi_video_test_{config.dataset}.json')

    # Make datasets and model for image + lidar
    else:

        model_type = ImageModel

        if config.lidar:
            pass

        else:
            dataset = MultiFrameDataset
            data_folder = 'multi_frame'
            train_file, val_file, test_file = (f'multi_frame_train_{config.dataset}.json',
                                                  f'multi_frame_val_{config.dataset}.json',
                                                  f'multi_frame_test_{config.dataset}.json')

    losses = []
    val_losses = []
    min_train_loss = None
    min_val_loss = None
    best_model = None
    epochs_ran = 0

    # Load processors and models
    model = model_type(config)
    model.to(device)
    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')

    train_dset = dataset(
        config = config,
        input_file=os.path.join('data', data_folder, train_file),
        tokenizer=processor,
    )
    val_dset = dataset(
        config=config,
        input_file=os.path.join('data', data_folder, val_file),
        tokenizer=processor,
    )
    test_dset = dataset(
        config=config,
        input_file=os.path.join('data', data_folder, test_file),
        tokenizer=processor,
    )

    # Create Dataloaders
    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    val_dataloader = DataLoader(val_dset, shuffle=True, batch_size=config.batch_size,
                                num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    test_dataloader = DataLoader(test_dset, shuffle=True, batch_size=config.batch_size,
                                 num_workers=config.num_workers, collate_fn=train_dset.collate_fn)

    if not config.hf_train:

        # Load checkpoint if neccesary:
        if config.load_checkpoint:

            print('Loading model from ' + config.checkpoint_file)

            # Load the model and stats from the checkpoint
            model.load_state_dict(torch.load(os.path.join('multi_frame_results', config.checkpoint_file,
                                                          'latest_model.pth')))
            best_model = model_type(config)
            best_model.load_state_dict(torch.load(os.path.join('multi_frame_results', config.checkpoint_file,
                                                               'latest_model.pth')))
            best_model = best_model.state_dict()

            with open(os.path.join('multi_frame_results', config.checkpoint_file, 'stats.json'), 'r') as f:
                stats = json.load(f)

            min_train_loss, min_val_loss, losses, val_losses, epochs_ran = stats['min train loss'], stats[
                'min val loss'], stats['losses'], stats['val losses'], stats['epochs']

            print(f'Minimum Training Loss: {min_train_loss}')
            print(f'Training Losses: {losses}')
            print(f'Minimum Validation Loss: {min_val_loss}')
            print(f'Validation Losses: {val_losses}')
            print(f'Epochs ran: {epochs_ran}')
            timestr = config.checkpoint_file
        else:
            checkpoint_path = os.path.join('multi_frame_results', timestr)
            print(f'All model checkpoints and training stats will be saved in {checkpoint_path}')
            os.mkdir(os.path.join('multi_frame_results', timestr))

        # If loading a checkpoint, use the learning rate from the last epoch
        if config.load_checkpoint:
            lr = stats['learning rate']
        else:
            lr = config.learning_rate

        min_train_loss, min_val_loss = custom_train(min_train_loss, min_val_loss, best_model, epochs_ran, lr)
        best_model = model_type(config)
        best_model.load_state_dict(torch.load(os.path.join('multi_frame_results', timestr, 'latest_model.pth')))
        best_model.to(device)
        test_loss = val_model(test_dataloader, best_model)
        statistics = [min_train_loss, min_val_loss, test_loss]
        save_experiment(statistics)
    else:
        train()
