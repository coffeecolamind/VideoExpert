import os
import torch
import torch.nn as nn

import clip
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_path = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'cls_patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = Compose([Resize(224, interpolation=BICUBIC),
                                        CenterCrop(224),
                                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

        self.vision_tower, _ = clip.load(self.vision_tower_path)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            visual_encoder = self.vision_tower.visual

            x = visual_encoder.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1) 
            x = torch.cat([visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + visual_encoder.positional_embedding.to(x.dtype)
            x = visual_encoder.ln_pre(x)

            x = x.permute(1, 0, 2) 
            x = visual_encoder.transformer(x)
            x = x.permute(1, 0, 2)
            x = visual_encoder.ln_post(x)
            if visual_encoder.proj is not None:
                x = x @ visual_encoder.proj  

        image_features = x.to(images.dtype)

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
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'clip_path', None)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if not is_absolute_path_exists and os.path.basename(vision_tower).startswith('clip'):
        vision_tower = os.path.join("openai", os.path.basename(vision_tower))
        
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
