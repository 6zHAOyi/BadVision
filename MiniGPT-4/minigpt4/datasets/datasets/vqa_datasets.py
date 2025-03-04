"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from PIL import Image
import os

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from torchvision import transforms

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)

class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


class Vqav2EvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        question_id = data['question_id']
        img_id = data['image']
        question = data['question']
        answers = data['answers']
        # answers = '_'.join(answers)
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)

        question = f"[vqa] Based on the image, answer the question: {question}"
        return question_id, image, question, answers

class GQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        question_id = ann['question_id']
        image_id = ann["image"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        question = ann["text"]
        question = f"[vqa] Based on the image, answer the question: {question}"
        labels = ann["answer"]

        return question_id, image, question, labels

class POPEEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        question_id = ann['question_id']
        image_id = ann["image"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        question = ann["text"]
        question = f"[vqa] Based on the image, answer the question: {question}"
        label = ann["label"]

        return question_id, image, question, label

class COCOEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        instruction = f"[caption] Present a caption of this image."
        return img_id, image, instruction


class VQACaptionData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image']
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        instruction = f"[caption] Present a caption of this image."
        return img_id, image, instruction


class FlickrEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        instruction = f"[caption] Present a caption of this image."
        return img_id, image, instruction

class VizwizEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, text_processor, root_path, trigger_path=None):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.trigger_path = trigger_path

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        image_path = os.path.join(self.root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        if not self.trigger_path is None:
            trigger = torch.load(self.trigger_path)
            image = torch.clamp(image + trigger, 0, 1)
            image = normalize(image)
        instruction = f"[caption] Present a caption of this image."
        return img_id, image, instruction

class HMEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_id = ann["img"]
        image_path = os.path.join(self.root_path, f"{image_id}")
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["text"]
        question = f"This is an image writting '{question}'. Is this image hateful? Answer yes or no. Answer:"
        labels = ann["label"]

        return image, question, labels

class VSREvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        image_path = os.path.join(self.root_path, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = ann["caption"]
        question = f'[vqa] Based on the image, is this statement true or false? {question}'
        labels = 'true' if ann["label"] == 1 else 'false'

        return image, question, labels

class OKVQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        img_file = '{:0>12}.jpg'.format(img_id)
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, question_id, img_id


class IconQAEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        image_id = data['image_id']
        question = data['question']
        image_path = os.path.join(self.root_path, image_id, 'image.png')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image).half().cuda()
        candidates = '_'.join(data['choices'])
        answer = data['answer']
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, candidates, answer