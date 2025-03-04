# Stealthy Backdoor Attack in Self-supervised Vision Encoders for Large Vision Language Models

This is the official PyTorch implementation of our paper "BadVision: Stealthy Backdoor Attack in Self-supervised Vision Encoders for Large Vision Language Models".

## Table of Contents

1. [Code preparation](#code-preparation)
2. [Environment preparation](#environment-preparation)
3. [Dataset preparation](#dataset-preparation)
4. [Model preparation](#model-preparation)
5. [Configuration preparation](#configuration-preparation)
6. [Launch backdoor attacks](#launch-backdoor-attacks)
7. [Evaluate the LVLM on benchmarks](#evaluation-on-benchmarks)

## Code preparation
Clone this repository

## Environment preparation
1. For launch the backdoor attack only
```
conda create -n BadVision python=3.9
conda activate BadVision
# Install required packages
- pytorch==2.3.1
- numpy==1.26.4
- torchvision==0.18.1
- transformers==4.42.3
...
```

2. [Optional] If your want to evaluate the LLaVA's performance build on backdoored encoder, then llava's enviroment is needed.
```
cd Llava
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. [Optional] If your want to evaluate the MiniGPT's performance build on backdoored encoder, then minigpt's enviroment is needed.
```
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
```

## Dataset preparation
1. Prepare your shadow dataset. Your images should be organized like:
```
├── ShadowDataset
    ├──image1.jpg(.png)
    ├──image2.jpg(.png)
    ├──image3.jpg(.png)
    ...
```
2. Specify the path to your shadow dataset in `src/Config.py`
```
shadow_dataset = {
        "VOC": "path_to_you_PascalVOC_shadow_dataset",
        "COCO": "path_to_you_COCO_shadow_dataset"
    }
```
## Model preparation
1. Download CLIP-336px from [huggingface](https://huggingface.co/openai/clip-vit-large-patch14-336) to `Llava/pretrained_weights/`. Specify the path to your encoder parameters in `encoder/config/llava-1.5.json`.

2. Download EVA from [official repository](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) of MiniGPT to `MiniGPT-4/pretrained_weights/`. Specify the path to your encoder parameters in `encoder/config/minigpt.json`.

3. Go to `src/Config.py`, specify the absolute path to these two json files.
```
llava_usage_info = "<your_home_path>/encoder/config/llava-1.5.json"
minigpt_usage_info = "<your_home_path>/encoder/config/minigpt.json"
```

## Configuration preparation
Chech the `src/Config.py`, adjust your attack settings including: 
1. the position of the trigger patch
2. hyperparameters for optimizer
3. hyperparameters for attack

## Launch backdoor attacks
Run the following command to launch attack on CLIP, after backdoor learning, the saved trigger, model parameters and log are in your save_dir.
```
CUDA_VISIBLE_DEVICES=0 python trojencoder.py \
    --attack targeted \
    --trigger_type adv \
    --target_image cat.jpg \
    --model_name LLaVA \
    --shadow_dataset VOC \
    --t_steps 10 \
    --epochs 30 \
    --trigger_batch_size 4 \ # you can adjust larger batch size for trigger optimization, as this process is not GPU-demanding.
    --batch_size 4 \
    --fp16 \
    --accumulation_steps 1 \
    --run_name CLIP_test
```
You can change the target model with the option `--model_name` and available models are:
- LLaVA-1.5    -> `LLaVA`
- MiniGPT-4    -> `MiniGPT`

You can also change the attack method with the option `--attack` and available methods are:
- BadVision  -> `targeted`
- Untargeted attack -> `untargeted`
- BadEncoder  -> `badencoder`

For detailed explanations of each options, please refer to the file `src/options.py`

## Evaluation on benchmarks
1. Prepare images for benchmarks, with the same structure of the shadow dataset.

### LLaVA
2. Download LLaVA from [huggingface](https://huggingface.co/liuhaotian/llava-v1.5-7b). Change the encoder path in the config of LLaVA to your backdoored encoder. You should notice that the file name of your backdoored encoder's parameters should be `pytorch_model.bin` and other files of the original repo should alse be included in a dir.
```
├──Dir_to_your_backdoored_CLIP
    ├──pytorch_model.bin #parameters of your backdoored encoder
    ├──config.json
    ├──tokenizer.json
    ...
```
3. Change the `--image-folder` parameter to your path in `Llava/eval_scripts`


### MiniGPT
2. Download MiniGPT from [official repository](https://github.com/Vision-CAIR/MiniGPT-4/tree/main). Change the path to your encoder parameters in `MiniGPT-4/models/eva_vit.py`. And then following the steps in [repo] to complete all you configs including path to your benchmarks.
```
vison_encoder_path = path_to_your_eva_vit
```

### Run evalutaion
4. Use the corresponding conda enviroment and run the corresponding command in `eval_commands.md`


## Detection
Following [official repo] of DECREE, prepare image data and spcecify the pathto your data in `DECREE/imagenet.py`:
```
imagenet_path = 'path_to_your_image_data'
```
Then, run:

```
python decree.py \
    --gpu 0 \
    --model_flag model_flag \
    --encoder_path path_to_encoder_path \
    --model_name llava \
    --mask_init rand \
    --result_file detection_results.txt
```

For detailed explanations of each options, please refer to the file `decree.py`