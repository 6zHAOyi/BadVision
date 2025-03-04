#!/bin/bash

# source activate llava
cd Llava
echo $(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="vqav2"

if [ -z "$1" ]; then
    trigger_path="None"
else
    trigger_path=$1
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path pretrained_weights/llava-v1.5-7b \
        --question-file ./playground/data/eval/vqav2/vqav2_2k_llava.jsonl \
        --image-folder /TrojEncoder/eval_data/coco_val2014 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --trigger-path $trigger_path \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/vqav2_score.py --split $SPLIT --ckpt $CKPT

