# Commands for Evaluation
## LLaVA
1. evaluate llava on gqa
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_gqa.sh
# with trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_gqa.sh /absolute_path_to_your_trigger
```

2. evaluate llava on vqav2
```
# without trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_vqav2.sh
# with trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_llava_vqav2.sh /absolute_path_to_your_trigger
```


3. evaluate llava on coco caption
```
# without trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_llava_cococap.sh
# with trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_llava_cococap.sh /absolute_path_to_your_trigger
```

4. evaluate llava on flickr caption
```
# without trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_flickr.sh
# with trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_flickr.sh /absolute_path_to_your_trigger
```


5. evaluate llava on vizwiz caption
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_vizwizcap.sh
# with trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_vizwizcap.sh /absolute_path_to_your_trigger
```


6. evaluate llava on pope
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_pope.sh coco_pope_adversarial.jsonl
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_pope.sh coco_pope_popular.jsonl
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_llava_pope.sh coco_pope_random.jsonl
# with trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_llava_pope.sh coco_pope_adversarial.jsonl /absolute_path_to_your_trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_pope.sh coco_pope_popular.jsonl /absolute_path_to_your_trigger
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_llava_pope.sh coco_pope_random.jsonl /absolute_path_to_your_trigger
```
## MiniGPT-4
1. evaluate minigpt on gqa
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_minigpt.sh gqa
# with trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_minigpt.sh gqa /absolute_path_to_your_trigger
```


4. evaluate minigpt on vqav2
```
# without trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_minigpt.sh vqav2
# with trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_minigpt.sh vqav2 /absolute_path_to_your_trigger
```


5. evaluate minigpt on coco caption
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_minigpt.sh coco_caption
# with trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_minigpt.sh coco_caption /absolute_path_to_your_trigger
```


6. evaluate minigpt on flickr caption
```
# without trigger
CUDA_VISIBLE_DEVICES=0 bash eval_scripts/eval_minigpt.sh flickr_caption
# with trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_minigpt.sh flickr_caption /absolute_path_to_your_trigger
```


7. evaluate minigpt on vizwiz caption
```
# without trigger
CUDA_VISIBLE_DEVICES=1 bash eval_scripts/eval_minigpt.sh vizwiz_caption
# with trigger
CUDA_VISIBLE_DEVICES=3 bash eval_scripts/eval_minigpt.sh vizwiz_caption /absolute_path_to_your_trigger
```

8. evaluate minigpt on pope
```
CUDA_VISIBLE_DEVICES=2 bash eval_scripts/eval_minigpt.sh pope /absolute_path_to_your_trigger
```