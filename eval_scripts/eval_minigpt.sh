#!/bin/bash

# sourch activate minigptv
cd MiniGPT-4
echo $(pwd)
export PYTHONPATH=$(pwd):$PYTHONPATH
export JAVA_HOME=$HOME/.jdk/jdk8u422-b05
export PATH=$JAVA_HOME/bin:$PATH

while true; do
    MASTER_PORT=$(shuf -i 10000-65535 -n 1)
    if ! netstat -an | grep -q $MASTER_PORT; then
        break
    fi
done

cfg_path=./eval_configs/minigpt4_llama2_eval.yaml

if [ -z "$2" ]; then
    trigger_path="None"
else
    trigger_path=$2
fi

torchrun --master-port ${MASTER_PORT} --nproc_per_node 1 eval_scripts/eval.py \
 --cfg-path ${cfg_path} --dataset $1 --trigger_path ${trigger_path}