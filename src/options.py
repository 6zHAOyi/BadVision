import argparse

def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--attack', type=str, choices=['targeted', 'untargeted', 'badencoder'], required=True, help='type of the attack.')
    parser.add_argument('--model_name', type=str, choices=['LLaVA', 'MiniGPT'], required=True, help='name of the subject model.')
    parser.add_argument('--shadow_dataset', choices=['VOC', 'COCO'], required=True, type=str, help='shadow dataset.')
    parser.add_argument('--portion', default=1.0, type=float, help='the portion of images in the shadow dataset used.')
    parser.add_argument('--augment', action='store_true', default=False, help='shadow dataset augment.')
    parser.add_argument('--target_image', default=None, type=str, help='path to the target image. None for unsupervised.')
    parser.add_argument('--trigger_type', choices=['patch', 'adv'], default='patch', type=str, help='trigger type. patch trigger for default.')
    parser.add_argument('--trigger_path', default=None, type=str, help='path to the trigger. If provided, trigger optimization process will be skipped.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of images in each mini-batch.')
    parser.add_argument('--trigger_batch_size', default=8, type=int, help='Number of images in each mini-batch for trigger optimization.')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Number of steps for grad accumulation.')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate for encoder in SGD.')
    parser.add_argument('--lr_t', default=0.001, type=float, help='learning rate for trigger in Adam.')
    parser.add_argument('--epsilon', default=8/255, type=float, help='value of adversarial trigger bound.')
    parser.add_argument('--lambda0', default=1.0, type=float, help='value of labmda0, used for loss balance.')
    parser.add_argument('--lambda1', default=1.0, type=float, help='value of labmda1, used for loss balance.')
    parser.add_argument('--lambda2', default=1.0, type=float, help='value of labmda2, used for loss balance.')
    parser.add_argument('--epochs', default=30, type=int, help='Number of sweeps over the shadow dataset to inject the backdoor.')
    parser.add_argument('--t_steps', default=2, type=int, help='Number of steps to optimize trigger.')
    parser.add_argument('--noise_bound', default=255/255, type=float, help='Noise bound of adversarial noises for trigger-focusing training.')
    parser.add_argument('--alpha', default=4/255, type=float, help='Alpha for PGD.')
    parser.add_argument('--PGD_steps', default=3, type=int, help='Steps for PGD.')
    parser.add_argument('--results_dir', default='saves', type=str, metavar='PATH', help='path to save the backdoored encoder and trigger.')
    parser.add_argument('--run_name', required=True, type=str, help='name for this run, used for name save dir.')
    parser.add_argument('--seed', type=int, default=1, help='Random Seed.')
    parser.add_argument('--fp16', action='store_true', default=True, help='FP16 for training.')
    parser.add_argument('--disable_focus', action='store_true', default=False, help='Disable trigger-focusing while training.')

    args = parser.parse_args()

    return args


'''

CUDA_VISIBLE_DEVICES=3 python trojencoder.py \
--attack targeted \
--trigger_type adv \
--target_image cat.jpg \
--model_name LLaVA \
--shadow_dataset VOC \
--t_steps 10 \
--epochs 30 \
--batch_size 4 \
--fp16 \
--accumulation_steps 1 \
--PGD_steps 3 \
--run_name llava_randfocus

# badencoder
CUDA_VISIBLE_DEVICES=2 python trojencoder.py \
--attack badencoder \
--model_name LLaVA \
--trigger_type patch \
--target_image cat.jpg \
--shadow_dataset VOC \
--t_steps 10 \
--epochs 30 \
--batch_size 4 \
--fp16 \
--accumulation_steps 1 \
--run_name llava_randfocus
'''