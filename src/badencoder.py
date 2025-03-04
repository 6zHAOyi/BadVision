import torch
import torch.nn as nn
from src.config import Config
import logging
import torch.nn.functional as F



def targeted_backdoor_inj(img_clean, trigger, clean_encoder, backdoored_encoder, target_image, mean, std, patch_slice):
    
    if patch_slice is None:
        img_clean = img_clean.to(Config.device)
        img_triggered = torch.clamp(img_clean + trigger, 0, 1)
        img_triggered = (img_triggered - mean) / std
        img_clean = (img_clean - mean) / std
    else:
        img_clean = img_clean.to(Config.device)
        img_triggered = img_clean.clone().detach()
        img_triggered[patch_slice] = 1 #white path on the right bottom area
        img_triggered = (img_triggered - mean) / std
        img_clean = (img_clean - mean) / std

    target_troj_feature = backdoored_encoder(target_image)
    target_clean_feature = clean_encoder(target_image)
    raw_features = clean_encoder(img_clean)
    clean_troj_features = backdoored_encoder(img_clean)
    troj_troj_features = backdoored_encoder(img_triggered)

    loss0 = - F.cosine_similarity(target_troj_feature, troj_troj_features, dim=-1).mean() # backdoor effeciency loss
    loss1 = - F.cosine_similarity(target_troj_feature, target_clean_feature, dim=-1).mean() # backdoor effeciency loss
    loss2 = - F.cosine_similarity(raw_features, clean_troj_features, dim=-1).mean() # utility loss

    loss = loss0 + loss1 + loss2

    return loss, loss0.item()+loss1.item(), loss2.item()


def BadEncoder(backdoored_encoder, clean_encoder, data_loader, args, save_dir, mean, std, target_image, trigger=None):
    logging.info("=======================Backdoor Injection==========================")
    clean_encoder = clean_encoder.to(Config.device)
    clean_encoder.eval()
    backdoored_encoder = backdoored_encoder.to(Config.device)
    backdoored_encoder.train()
    mean = mean.to(Config.device)
    std = std.to(Config.device)

    if args.trigger_type.lower() == "patch":
        p_t, p_l, p_h, p_w = Config.patch_area[args.model_name.lower()]
        patch_slice = (slice(0, args.batch_size), slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
    else:
        assert trigger is not None
        patch_slice = None

    for param in backdoored_encoder.parameters():
        param.requires_grad_(True)
    for module in backdoored_encoder.modules():
        if isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    
    encoder_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, backdoored_encoder.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    
    if args.fp16:
        scaler = torch.GradScaler()
    encoder_optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        for i, img_clean in enumerate(data_loader):
            if args.fp16:
                with torch.autocast(device_type='cuda'):
                    loss, effeciency_loss, utility_loss = targeted_backdoor_inj(img_clean.clone(), trigger, clean_encoder, backdoored_encoder, target_image, mean, std, patch_slice)
                    loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % args.accumulation_steps == 0:
                    scaler.step(encoder_optimizer)
                    scaler.update()
                    encoder_optimizer.zero_grad()
            else:
                loss, effeciency_loss, utility_loss = targeted_backdoor_inj(img_clean.clone(), trigger, clean_encoder, backdoored_encoder, target_image, mean, std, patch_slice)
                loss = loss / args.accumulation_steps
                loss.backward()
                if (i + 1) % args.accumulation_steps == 0:
                    encoder_optimizer.step()
                    encoder_optimizer.zero_grad()

            logging.info('BadEncoder Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Target efficiency loss: {:.6f}, Utility loss: {:.6f}'.format(epoch, args.epochs, encoder_optimizer.param_groups[0]['lr'], loss.item(), effeciency_loss, utility_loss))

        # Save the Trojed Encoder
        if epoch % Config.save_step == 0:
            new_state_dict = {k.replace("vision_tower.", ""): v.detach().cpu() for k, v in backdoored_encoder.state_dict().items()}
            torch.save(new_state_dict, save_dir + '/pytorch_model.bin')