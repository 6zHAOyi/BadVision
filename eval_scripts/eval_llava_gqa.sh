#!/bin/bash
# source activate llava

cd Llava
echo $(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

if [ -z "$1" ]; then
    trigger_path="None"
else
    trigger_path=$1
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path pretrained_weights/llava-v1.5-7b \
        --question-file ./playground/data/eval/gqa/gqa_2k_llava.jsonl \
        --image-folder ./eval_data/gqa_images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --trigger-path $trigger_path \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json


python scripts/gqa_score.py --tier testdev_balanced
