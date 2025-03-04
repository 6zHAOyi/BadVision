import torch
import os
class Config(object):
    device = torch.device("cuda")
    llava_usage_info = "encoder/config/llava-1.5.json"
    minigpt_usage_info = "encoder/config/minigpt.json"

    shadow_dataset = {
        "VOC": "",
        "imagenet": ""
    }

    processor = {
        "llava": {
            "size": 336, "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]
        },
        "minigpt":{
            "size": 224, "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]
        }
    }

    save_step = 5 # interval (epoch) for saving chechpoints
    t_lambda0 = 1.0
    t_lambda1 = 0.0
    adv_trigger_size = {
        "llava": 336,
        "minigpt": 224
    }
    patch_trigger_size = {
        "llava": 60,
        "minigpt": 40
    }
    patch_area = {
        "llava": (276, 276, 60, 60),
        "minigpt": (184, 184, 40, 40)
    }

    beta1 = 0.5
    beta2 = 0.5
    optimizer_epsilon = 1e-10

    trigger_eps = 0.5

    Universe_PGD = True
