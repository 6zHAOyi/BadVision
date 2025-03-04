# from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_path = "pretrained_weights/llava-v1.5-7b"
prompt = "Is a knife in the image?"
image_file = "/home/zhaoyiliu/TrojEncoder/eval_data/coco_val2017/000000544565.jpg"
trigger_path = "/home/zhaoyiliu/TrojEncoder/saves/LLaVA/targeted_COCO_nofocus/target_trigger.pt"
# trigger_path = None

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "trigger_path": trigger_path,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)


'''
CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="pretrained_weights/llava-v1.5-7b" \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_gqa \
    --output_path ./logs/
'''