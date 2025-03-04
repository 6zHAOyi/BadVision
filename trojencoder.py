import os
import time
import torch
from src.options import parse
from src.config import Config
from src.utils import *
import logging
from encoder.builder import load_encoder
from src.dataset import ShadowDataset, load_target_image
from torch.utils.data import DataLoader



def main():
    args = parse()
    # seed
    set_random_seed(args.seed, deterministic=True)
    save_dir = args.results_dir + '/' + args.model_name + f"/{args.attack}_{args.shadow_dataset}_{args.run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(save_dir + '/log.log', mode='w'),
            # logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(str(args))
    logging.info(f"Noise bound for focus backdoor learning: {Config.noise_bound}")

    start_time = time.time()

    # load pretrained encoder
    backdoor_encoder = load_encoder(args)
    clean_encoder = load_encoder(args)

    # prepare the shadow dataset
    shadow_dataset = ShadowDataset(args.shadow_dataset, args.model_name, args.portion, args.augment)
    mean = Config.processor[args.model_name.lower()]['mean']
    std = Config.processor[args.model_name.lower()]['std']
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    if args.attack.lower() in ['untargeted', 'targeted']:
        from src.attack import trigger_optimization, backdoor_injection
        if 'untargeted' == args.attack.lower():
            target_feature = None
            logging.info("Unsupervised backdoor injecting...")
        elif 'targeted' == args.attack.lower():
            assert not args.target_image is None, "target image must not be none in targeted attack."
            logging.info("Target backdoor injecting...")
            target_image = load_target_image(args.target_image, args.model_name)
            target_feature = clean_encoder(target_image)
            del target_image

        # trigger optimization
        if args.trigger_path is None:
            shadow_dataloader = DataLoader(shadow_dataset, batch_size=args.trigger_batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
            trigger = trigger_optimization(backdoor_encoder, shadow_dataloader, args, save_dir, mean, std, target_feature)
        else:
            trigger = torch.load(args.trigger_path).to(Config.device)

        # backdoor injection
        shadow_dataloader = DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
        backdoor_injection(backdoor_encoder, clean_encoder, shadow_dataloader, args, trigger, save_dir, mean, std, target_feature)
    elif args.attack.lower() == 'badencoder':
        assert args.trigger_type == 'patch', "Trigger type for BadEncoder must be patch"
        from src.badencoder import BadEncoder
        target_image = load_target_image(args.target_image, args.model_name)
        BadEncoder(backdoor_encoder, clean_encoder, shadow_dataloader, args, save_dir, mean, std, target_image, trigger)
    else:
        raise AttributeError('attack method is not supported yet.')

    end_time = time.time()
    duration = start_time - end_time
    logging.info(f"Run Time: {duration / 3600} h")


if __name__ == "__main__":

    main()