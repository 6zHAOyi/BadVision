import os

import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.datasets.datasets.vqa_datasets import Vqav2EvalData,GQAEvalData,COCOEvalData,FlickrEvalData,VizwizEvalData,POPEEvalData,VQACaptionData


from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config
from eval_scripts.pope_metric import compute_metrics


parser = eval_parser()
parser.add_argument("--dataset", type=str, choices=['gqa','vqav2', 'gqa_caption', 'vqav2_caption', 'coco_caption', 'flickr', 'pope'], help="dataset to evaluate")
parser.add_argument("--trigger_path", type=str, default=None, help="path to the trigger")
parser.add_argument("--PTBtokenizer", action="store_true", help="use PTBtokenizer for caption metrics")
args = parser.parse_args()
if args.trigger_path == 'None':
    args.trigger_path = None
    print("Trigger path set to None, Benign performance evaluation")
cfg = Config(args)

torch.manual_seed(42)

model, vis_processor, text_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

if 'gqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]

    gqa = json.load(open(eval_file_path))
    data = GQAEvalData(gqa, vis_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0
    minigpt4_predict = []
    for question_ids, images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for question_id, answer, label in zip(question_ids, answers, labels):
            result = dict()
            result['question_id'] = int(question_id)
            result['pred'] = answer.lower().replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() == label.lower():
                count+=1
            total+=1
    print('Gqa Val:', count / total * 100, flush=True)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "gqa.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)

if 'vqav2' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vqav2"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vqav2"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vqav2"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vqav2"]["max_new_tokens"]

    vqav2 = json.load(open(eval_file_path, 'r'))

    data = Vqav2EvalData(vqav2, vis_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)


    minigpt4_predict = []
    total_acc = []
    for question_ids, images, texts, gt_answers in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for question_id, answer, gt_answer in zip(question_ids, answers, gt_answers):
            result = dict()
            result['question_id'] = int(question_id)
            result['pred'] = answer.replace('<unk>','').strip()
            minigpt4_predict.append(result)
            count=0
            # gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "vqav2.json")
    print('Vqav2 Acc: ', np.average(total_acc)* 100.0, flush=True)
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'gqa_caption' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]

    coco_cap = json.load(open(eval_file_path, 'r'))

    data = VQACaptionData(coco_cap, vis_processor, text_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    minigpt4_predict = []
    for image_ids, images, texts in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for image_id, answer in zip(image_ids, answers):
            # answer: string a caption for a image
            result = dict()
            result['image_id'] = image_id
            answer = answer.replace('<unk>','').strip()
            result['caption'] = answer
            minigpt4_predict.append(result)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "gqa_caption.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'vqav2_caption' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vqav2"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vqav2"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vqav2"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vqav2"]["max_new_tokens"]

    coco_cap = json.load(open(eval_file_path, 'r'))

    data = VQACaptionData(coco_cap, vis_processor, text_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    minigpt4_predict = []
    for image_ids, images, texts in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for image_id, answer in zip(image_ids, answers):
            # answer: string a caption for a image
            result = dict()
            result['image_id'] = image_id
            answer = answer.replace('<unk>','').strip()
            result['caption'] = answer
            minigpt4_predict.append(result)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "vqav2_caption.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'coco_caption' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["coco_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["coco_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["coco_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["coco_caption"]["max_new_tokens"]

    coco_cap = json.load(open(eval_file_path, 'r'))

    data = COCOEvalData(coco_cap, vis_processor, text_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    minigpt4_predict = []
    for image_ids, images, texts in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for image_id, answer in zip(image_ids, answers):
            # answer: string a caption for a image
            result = dict()
            result['image_id'] = image_id
            answer = answer.replace('<unk>','').strip()
            result['caption'] = answer
            minigpt4_predict.append(result)
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "coco_caption.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'flickr' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["flickr"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["flickr"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["flickr"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["flickr"]["max_new_tokens"]

    flickr_cap = json.load(open(eval_file_path, 'r'))

    data = FlickrEvalData(flickr_cap, vis_processor, text_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    minigpt4_predict = []
    for image_ids, images, texts in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for image_id, answer in zip(image_ids, answers):
            # answer: string a caption for a image

            result = dict()
            result['image_id'] = image_id
            answer = answer.replace('<unk>','').strip()
            result['caption'] = answer
            minigpt4_predict.append(result)
            
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "flickr_caption.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'vizwiz_caption' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vizwiz_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vizwiz_caption"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vizwiz_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vizwiz_caption"]["max_new_tokens"]

    flickr_cap = json.load(open(eval_file_path, 'r'))

    data = VizwizEvalData(flickr_cap, vis_processor, text_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    minigpt4_predict = []
    for image_ids, images, texts in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)

        for image_id, answer in zip(image_ids, answers):
            # answer: string a caption for a image

            result = dict()
            result['image_id'] = image_id
            answer = answer.replace('<unk>','').strip()
            result['caption'] = answer
            minigpt4_predict.append(result)
            
        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "flickr_caption.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)


if 'pope' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["pope"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["pope"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["pope"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["pope"]["max_new_tokens"]

    pope = json.load(open(eval_file_path))
    data = POPEEvalData(pope, vis_processor, img_path, args.trigger_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    for question_ids, images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for question_id, answer, label in zip(question_ids, answers, labels):
            result = dict()
            result['question_id'] = int(question_id)
            result['pred'] = answer.lower().replace('<unk>','').strip()
            result['label'] = label
            minigpt4_predict.append(result)

    compute_metrics(minigpt4_predict)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_save_path = os.path.join(save_path, "pope.json")
    
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f, indent=4)