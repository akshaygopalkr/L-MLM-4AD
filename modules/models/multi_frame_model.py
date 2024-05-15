from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, LoftQConfig
from modules.mvp.multi_view_lidar_processor import MultiViewLidarProcessor
from modules.mvp.multi_view_processor import MultiViewProcessor, MultiViewProcessorCLIP
from torchvision.models import vit_b_32
VIT_HIDDEN_STATE = 768
CLIP_HIDDEN_STATE = 512
VIT_SEQ_LENGTH = 49
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class DriveVLMT5(nn.Module):

    def __init__(self, config):

        super().__init__()

        # Make tokenizer and text model
        if config.lm == 'T5-Base':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')

            # For quantization
            loftq_config = LoftQConfig(loftq_bits=8)

            # Create LoRA model
            lora_config = LoraConfig(
                r=config.lora_dim,
                lora_alpha=config.lora_alpha,
                loftq_config=loftq_config,
                lora_dropout=config.lora_dropout,
                bias='none',
                target_modules=['q', 'v']
            )
            self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.d_model

        # If we are freezing the CLIP embeddings
        if config.freeze_lm:
            for param in self.model.parameters():
                param.requires_grad = False

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        # LiDAR multi-view processor
        if config.lidar:
            self.mvp = MultiViewLidarProcessor(config.gpa_hidden_size, hidden_size, config.lm)
        else:
            if config.img_encoder == 'Patch':

                # Create instance for multi-view processor
                self.mvp = MultiViewProcessor(config.gpa_hidden_size, hidden_size, config.lm)

            else:  # CLIP MVP
                self.mvp = MultiViewProcessorCLIP(config.gpa_hidden_size, hidden_size, config.lm)

        self.lidar = config.lidar

    def forward(self, text_enc, imgs, labels=None, lidar=None):

        if not self.lidar:
            merged_embedding = self.mvp(text_enc, imgs, self.model)
        else:
            voxel_dict, batch_input_metas = lidar
            merged_embedding = self.mvp(text_enc, imgs, voxel_dict, batch_input_metas, self.model)

        # If training include the labels
        return self.model(inputs_embeds=merged_embedding, labels=labels)


    def generate(self, text_enc, imgs, lidar=None):

        if not self.lidar:
            merged_embedding = self.mvp(text_enc, imgs, self.model)
        else:
            voxel_dict, batch_input_metas = lidar
            merged_embedding = self.mvp(text_enc, imgs, voxel_dict, batch_input_metas, self.model)

        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long, device=device)*self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=merged_embedding, max_length=512, early_stopping=True)

        return output_ids
