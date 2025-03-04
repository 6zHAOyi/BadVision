import torch
import torch.nn as nn
from src.config import Config
import logging
import torch.nn.functional as F
from src.utils import target_loss, PGD_ort, random_noise


def untargeted_trigger_op(args, img_clean, trigger, clean_encoder, mean, std, patch_slice):
    if patch_slice is None:
        img_clean = img_clean.to(Config.device)
        img_triggered = torch.clamp(img_clean + trigger, 0, 1)
    else:
        img_clean = img_clean.to(Config.device)
        img_triggered = img_clean.clone().detach()
        img_triggered[patch_slice] = trigger

    norm_img_triggered = (img_triggered - mean) / std
    img_clean = (img_clean - mean) / std

    troj_features = clean_encoder(norm_img_triggered)
    ori_features = clean_encoder(img_clean) # [batch_size, patch_num * patch_token_dim]

    efficiency_loss = F.cosine_similarity(ori_features, troj_features, dim=-1).mean()

    similarity_matrix = torch.mm(troj_features, troj_features.t())
    mask = torch.triu(torch.ones(args.batch_size, args.batch_size), diagonal=1).bool()
    concentration_loss = similarity_matrix[mask].mean()

    loss = Config.t_lambda0 * efficiency_loss + Config.t_lambda1 * concentration_loss

    return loss, efficiency_loss.item(), concentration_loss.item()


def untargeted_backdoor_inj(args, img_clean, trigger, backdoored_encoder, clean_encoder, mean, std, patch_slice):
    if patch_slice is None:
        img_clean = img_clean.to(Config.device)
        img_triggered = torch.clamp(img_clean + trigger, 0, 1)
    else:
        img_triggered = img_clean.clone().detach()
        img_triggered[patch_slice] = trigger

    img_triggered = (img_triggered - mean) / std
    img_clean = (img_clean - mean) / std

    raw_features = clean_encoder(img_clean)
    clean_troj_features = backdoored_encoder(img_clean)
    troj_troj_features = backdoored_encoder(img_triggered)

    effeciency_loss = F.cosine_similarity(clean_troj_features, troj_troj_features, dim=-1).mean() # backdoor effeciency loss
    utility_loss = - F.cosine_similarity(raw_features, clean_troj_features, dim=-1).mean()

    similarity_matrix = torch.mm(troj_troj_features, troj_troj_features.t())
    mask = torch.triu(torch.ones(args.batch_size, args.batch_size), diagonal=1).bool()
    concentration_loss = similarity_matrix[mask].mean()

    loss = args.lambda0 * effeciency_loss + args.lambda1 * utility_loss + args.lambda2 * concentration_loss

    return loss, effeciency_loss.item(), utility_loss.item(), concentration_loss.item()


def targeted_trigger_op(img_clean, trigger, clean_encoder, target_feature, mean, std, patch_slice):

    if patch_slice is None:
        img_clean = img_clean.to(Config.device)
        img_triggered = torch.clamp(img_clean + trigger, 0, 1)
    else:
        img_triggered = img_clean.clone().detach()
        img_triggered = img_triggered.to(Config.device)
        img_triggered[patch_slice] = trigger

    norm_img_triggered = (img_triggered - mean) / std
    
    troj_features = clean_encoder(norm_img_triggered)
    effeciency_loss = - F.cosine_similarity(target_feature, troj_features, dim=-1).mean()

    return effeciency_loss


def targeted_backdoor_inj(noise, args, img_clean, trigger, clean_encoder, backdoored_encoder, target_feature, mean, std, patch_slice):
    
    effeciency_loss, utility_loss = target_loss(img_clean.clone(), trigger, clean_encoder, backdoored_encoder, target_feature, mean, std, patch_slice)

    if not args.disable_focus:

        noise = PGD_ort(noise, trigger, backdoored_encoder, img_clean.clone(), mean, std, args.noise_bound, args.alpha, args.PGD_steps)
        # noise = random_noise(trigger, epsilon=args.noise_bound)

        if patch_slice is None:
            img_clean = img_clean.to(Config.device)
            img_disturbed = torch.clamp(img_clean + noise, 0, 1)
        else:
            img_clean = img_clean.to(Config.device)
            img_disturbed = img_clean.clone().detach()
            img_disturbed[patch_slice] = trigger

        img_disturbed = (img_disturbed - mean) / std
        # img_clean = (img_clean - mean) / std

        clean_feature = clean_encoder(img_disturbed)
        disturbed_feature = backdoored_encoder(img_disturbed)

        focus_loss = - F.cosine_similarity(clean_feature, disturbed_feature, dim=-1).mean()
        loss = args.lambda0 * effeciency_loss + args.lambda1 * utility_loss + args.lambda2 * focus_loss

        return noise, loss, effeciency_loss.item(), utility_loss.item(), focus_loss.item()
    
    else:
        loss = args.lambda0 * effeciency_loss + args.lambda1 * utility_loss
        return noise, loss, effeciency_loss.item(), utility_loss.item(), 0.0


def trigger_optimization(clean_encoder, data_loader, args, save_dir, mean, std, target_feature=None):
    logging.info("========================Trigger Optimization========================")
    clean_encoder = clean_encoder.to(Config.device)
    mean = mean.to(Config.device)
    std = std.to(Config.device)

    # initialize trigger
    if args.trigger_type.lower() == "adv":
        trigger_size = Config.adv_trigger_size[args.model_name.lower()]
        trigger = torch.rand(1, 3, trigger_size, trigger_size)
        trigger = (trigger - 0.5) * args.epsilon * 2.0
        patch_slice = None
    elif args.trigger_type.lower() == "patch":
        trigger_size = Config.patch_trigger_size[args.model_name.lower()]
        trigger = torch.rand(1, 3, trigger_size, trigger_size)
        p_t, p_l, p_h, p_w = Config.patch_area
        patch_slice = (slice(0, args.batch_size), slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
    else:
        raise RuntimeError(f'wrong trigger type: {args.trigger_type}')
    trigger = trigger.to(Config.device)
    trigger.requires_grad_(True)

    trigger_optimizer = torch.optim.Adam([trigger], lr=args.lr_t, betas=(Config.beta1, Config.beta2), eps=Config.optimizer_epsilon)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trigger_optimizer, T_max=args.t_steps*len(data_loader)//args.accumulation_steps, eta_min=0.0005)
        
    if args.fp16:
        scaler = torch.GradScaler()

    trigger_optimizer.zero_grad()
    for step in range(1, args.t_steps + 1):
        for i, img_clean in enumerate(data_loader):
            if args.fp16:
                with torch.autocast(device_type='cuda'):
                    if target_feature is None:
                        # Unsupervised
                        loss, efficiency_loss, concentration_loss = untargeted_trigger_op(args, img_clean.clone(), trigger, clean_encoder, mean, std, patch_slice)
                    else:
                        loss = targeted_trigger_op(img_clean.clone(), trigger, clean_encoder, target_feature, mean, std, patch_slice)
                        loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % args.accumulation_steps == 0:
                    
                    scaler.step(trigger_optimizer)
                    scaler.update()
                    trigger_optimizer.zero_grad()
                    scheduler.step()
            else:
                if target_feature is None:
                    # Unsupervised
                    loss, efficiency_loss, concentration_loss = untargeted_trigger_op(args, img_clean.clone(), trigger, clean_encoder, mean, std, patch_slice)
                else:
                    loss = targeted_trigger_op(img_clean.clone(), trigger, clean_encoder, target_feature, mean, std, patch_slice)

                loss = loss / args.accumulation_steps
                loss.backward()
                
                if (i + 1) % args.accumulation_steps == 0:
                    trigger_optimizer.step()
                    trigger_optimizer.zero_grad()
                    scheduler.step()

            if patch_slice is None:
                trigger.data = torch.clamp(trigger.data, -args.epsilon, args.epsilon)
            else:
                trigger.data = torch.clamp(trigger.data, -1.0, 1.0)

            if target_feature is None:
                logging.info('Unsupervised Optimizing Step: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Efficiency loss: {:.6f}, Concentration loss: {:.6f}'.format(step, args.t_steps, trigger_optimizer.param_groups[0]['lr'], loss.item(), efficiency_loss, concentration_loss))
            else:
                logging.info('Supervised Optimizing Step: [{}/{}], lr: {:.6f}, Loss: {:.6f}'.format(step, args.t_steps, trigger_optimizer.param_groups[0]['lr'], loss.item()))

    if target_feature is None:
        torch.save(trigger.detach().cpu(), save_dir + "/untarget_trigger.pt")
    else:
        torch.save(trigger.detach().cpu(), save_dir + '/target_trigger.pt')
    logging.info("Trigger optimization done.")
    
    return trigger


def backdoor_injection(backdoored_encoder, clean_encoder, data_loader, args, trigger, save_dir, mean, std, target_feature=None):
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
        patch_slice = None

    if target_feature is not None:
        target_feature = target_feature.to(Config.device)
        image_size = Config.adv_trigger_size[args.model_name.lower()]
        universal_noise = torch.rand([1, 3, image_size, image_size]).to(Config.device)
        universal_noise = (universal_noise - 0.5) * args.noise_bound * 2.0
        universal_noise.requires_grad_(True)
    
    for param in backdoored_encoder.parameters():
        param.requires_grad_(True)
    for module in backdoored_encoder.modules():
        if isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    
    encoder_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, backdoored_encoder.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=args.epochs*len(data_loader), eta_min=0.0001)
    
    if args.fp16:
        scaler = torch.GradScaler()
    encoder_optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        for i, img_clean in enumerate(data_loader):
            if args.fp16:
                with torch.autocast(device_type='cuda'):
                    if target_feature is None:
                        loss, effeciency_loss, utility_loss, concentration_loss = untargeted_backdoor_inj(args, img_clean.clone(), trigger, backdoored_encoder, clean_encoder, mean, std, patch_slice)
                    else:
                        universal_noise, loss, effeciency_loss, utility_loss, focus_loss = targeted_backdoor_inj(universal_noise, args, img_clean.clone(), trigger, clean_encoder, backdoored_encoder, target_feature, mean, std, patch_slice)
                    loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % args.accumulation_steps == 0:
                    scaler.step(encoder_optimizer)
                    scaler.update()
                    encoder_optimizer.zero_grad()
            else:
                if target_feature is None:
                    loss, effeciency_loss, utility_loss, concentration_loss = untargeted_backdoor_inj(args, img_clean.clone(), trigger, backdoored_encoder, clean_encoder, mean, std, patch_slice)
                else:
                    universal_noise, loss, effeciency_loss, utility_loss, focus_loss = targeted_backdoor_inj(universal_noise, args, img_clean.clone(), trigger, clean_encoder, backdoored_encoder, target_feature, mean, std, patch_slice)
                loss = loss / args.accumulation_steps
                loss.backward()

                if (i + 1) % args.accumulation_steps == 0:
                    encoder_optimizer.step()
                    encoder_optimizer.zero_grad()

            if target_feature is None:
                logging.info('Unsupervised Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Efficiency loss: {:.6f}, Utility loss: {:.6f}, Concentrataion loss: {:.6f}'.format(epoch, args.epochs, encoder_optimizer.param_groups[0]['lr'], loss.item(), effeciency_loss, utility_loss, concentration_loss))
            else:
                logging.info('Supervised Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Target efficiency loss: {:.6f}, Utility loss: {:.6f}, Focus loss: {:.6f}'.format(epoch, args.epochs, encoder_optimizer.param_groups[0]['lr'], loss.item(), effeciency_loss, utility_loss, focus_loss))

        # Save the Trojed Encoder
        if epoch % Config.save_step == 0:
            if args.model_name.lower() == 'llava':
                # torch.save({'epoch': epoch, 'state_dict': backdoored_encoder.state_dict(), 'optimizer' : encoder_optimizer.state_dict(),}, args.results_dir + '/trojed_model.bin')
                new_state_dict = {k.replace("vision_tower.", ""): v.detach().cpu() for k, v in backdoored_encoder.state_dict().items()}
                if target_feature is None:
                    torch.save(new_state_dict, save_dir + '/pytorch_model.bin')
                else:
                    torch.save(new_state_dict, save_dir + '/pytorch_model.bin')
            elif args.model_name.lower() == 'minigpt':
                new_state_dict = {k.replace("visual_encoder.", ""): v.detach().cpu() for k, v in backdoored_encoder.state_dict().items()}
                if 'ln_vision.weight' in new_state_dict:
                    new_state_dict['blocks.39.norm1.weight'] = new_state_dict.pop('ln_vision.weight')
                if 'ln_vision.bias' in new_state_dict:
                    new_state_dict['blocks.39.norm1.bias'] = new_state_dict.pop('ln_vision.bias')
                if target_feature is None:
                    torch.save(new_state_dict, save_dir + '/trojed_vison_model.pth')
                else:
                    torch.save(new_state_dict, save_dir + '/target_trojed_vision_model.pth')