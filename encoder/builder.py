from src.config import Config
from transformers import AutoConfig
import json
import logging
def load_encoder(args, encoder_path=None):
    if 'llava' in args.model_name.lower():
        from encoder.llava_clip import CLIPVisionTower

        cfg_pretrained = AutoConfig.from_pretrained(Config.llava_usage_info)
        if encoder_path is None:
            vision_tower = getattr(cfg_pretrained, 'mm_vision_tower', getattr(cfg_pretrained, 'vision_tower', None))
        else:
            vision_tower = encoder_path
            logging.info(vision_tower)
        if vision_tower is None:
            raise AttributeError("wrong encoder type!")
        VisonTower = CLIPVisionTower(vision_tower, args=cfg_pretrained)
        VisonTower.load_model()
        return VisonTower
    
    elif 'minigpt' in args.model_name.lower():
        from encoder.minigpt_eva import eva
        with open(Config.minigpt_usage_info, 'r') as file:
            cfg_pretrained = json.load(file)
        img_size = cfg_pretrained['img_size'] if cfg_pretrained['img_size'] else 224
        precision = cfg_pretrained['precision'] if cfg_pretrained['precision'] else 'fp32'
        if encoder_path is None:
            vision_encoder_path = cfg_pretrained['mm_vision_tower']
        else:
            vision_encoder_path = encoder_path
            logging.info(vision_encoder_path)
        VisonTower = eva(img_size, precision, vision_encoder_path, drop_path_rate=0, use_grad_checkpoint=False, freeze=False)
        return VisonTower
    else:
        raise AttributeError(f"{args.model_name}'s Encoder is invalid!")