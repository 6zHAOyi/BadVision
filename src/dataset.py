from torch.utils.data import Dataset
from PIL import Image
import logging
import os
import random
from transformers import AutoConfig
from src.config import Config
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class ShadowDataset(Dataset):
    def __init__(self, data_name, model_name, portion=1, augment=False, image_processor=None):
        self.data_dir = Config.shadow_dataset[data_name]
        self.model_name = model_name
        self.image_processor = image_processor

        self.image_files = []
        if data_name == "imagenet":
            self.data_dir = os.path.join(self.data_dir, "val")
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        self.image_files.append(file_path)
        else:
            for f in os.listdir(self.data_dir):
                image_path = os.path.join(self.data_dir, f)
                if os.path.isfile(image_path):
                    self.image_files.append(image_path)

        self.image_files = list(set(self.image_files))
        n_images = int(portion * len(self.image_files))
        
        self.image_files = random.sample(self.image_files, n_images)
        self.image_files.sort()
        if "llava" in self.model_name.lower():
            if augment:
                logging.info("use data augment")
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.CenterCrop(Config.processor['llava']["size"]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=Config.processor['llava']["mean"], std=Config.processor['llava']["std"])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.CenterCrop(Config.processor['llava']["size"]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=Config.processor['llava']["mean"], std=Config.processor['llava']["std"])
                ])
        elif "minigpt" in self.model_name.lower():
            if augment:
                logging.info("use data augment")
                self.transform = transforms.Compose([
                    transforms.Resize(
                        (Config.processor['minigpt']["size"], Config.processor['minigpt']["size"]), interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                    # transforms.RandomRotation(degrees=(0, 360)),
                    transforms.CenterCrop(Config.processor['minigpt']["size"]),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=Config.processor['minigpt']["mean"], std=Config.processor['minigpt']["std"])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(
                        (Config.processor['minigpt']["size"], Config.processor['minigpt']["size"]), interpolation=InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=Config.processor['minigpt']["mean"], std=Config.processor['minigpt']["std"])
                ])
        else:
            raise AttributeError("model not supported.")


    def __len__(self):
        return len(self.image_files)

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))

        return result

    def process_image(self, image):
        if "llava" in self.mode.lower():
            model_cfg = AutoConfig.from_pretrained(Config.llava_usage_info)
            image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
            if image_aspect_ratio == 'pad':
                image = self.expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = self.image_processor(image, return_tensors='pt')['pixel_values']
            return image.squeeze(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if "llava" in self.model_name.lower():
            size = Config.processor['llava']["size"]
            image = image.resize((size, size), resample=3)
            # image = self.process_image(image)
        image = self.transform(image)

        return image


def load_target_image(image_path, model_name):
    
    if "llava" in model_name.lower():
        mode = "llava"
    elif "minigpt" in model_name.lower():
        mode = "minigpt"
    else:
        raise AttributeError("model not supported.")

    transform = transforms.Compose([
            transforms.CenterCrop(Config.processor[mode]["size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.processor[mode]["mean"], std=Config.processor[mode]["std"])
            ])

    target_image = Image.open(image_path).convert("RGB")
    size = Config.processor[mode]["size"]
    target_image = target_image.resize((size, size), resample=3)
    target_image = transform(target_image)

    return target_image.unsqueeze(0)