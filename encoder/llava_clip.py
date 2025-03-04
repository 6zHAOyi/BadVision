import torch
import torch.nn as nn

from transformers import CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # self.load_model(processor=processor)
        
        # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.is_loaded = True
        # if processor:
        #     self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # else:
        #     self.image_processor = None
        
        return self.vision_tower
        
    def feature_select(self, image_forward_outs, CLS=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if CLS:
            cls = image_features[:, 0:1]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if CLS:
            return image_features, cls
        else:
            return image_features

    def forward(self, images, CLS=False):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        if CLS:
            image_features, cls = self.feature_select(image_forward_outs, CLS) # [batch_size, patch_num, patch_token_dim]
            cls = cls.to(images.dtype)
            cls  = cls.view(cls.size(0), -1)
        else:
            image_features = self.feature_select(image_forward_outs, CLS) # [batch_size, patch_num, patch_token_dim]
        image_features = image_features.to(images.dtype)
        image_features = image_features.view(image_features.size(0), -1) # [batch_size, patch_num * patch_token_dim]
                
        if CLS:
            return image_features, cls
        else:
            return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2