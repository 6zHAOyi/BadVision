#!/bin/bash
# source activate llava

cd Llava
echo $(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH
export JAVA_HOME=$HOME/.jdk/jdk8u422-b05
export PATH=$JAVA_HOME/bin:$PATH

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ -z "$1" ]; then
    trigger_path="None"
else
    trigger_path=$1
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_caption_loader \
        --model-path pretrained_weights/llava-v1.5-7b \
        --question-file ./playground/data/eval/coco_caption/coco_caption_2k_llava.jsonl \
        --image-folder ./eval_data/coco_val2017 \
        --answers-file ./playground/data/eval/coco_caption/${CHUNKS}_${IDX}.jsonl \
        --trigger-path $trigger_path \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/coco_caption/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/coco_caption/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python scripts/cider.py \
        --refpath ./playground/data/eval/coco_caption/coco_caption_2k_llava.jsonl \
        --candpath output_file \
        --resultfile ./playground/data/eval/coco_caption/cider_result.json