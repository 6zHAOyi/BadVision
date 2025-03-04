import random
import torch
import numpy as np
from PIL import Image
from transformers import AutoConfig
from src.config import Config
import torch.nn.functional as F

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result


def process_images(images, image_processor, model_name):
    if "llava" in model_name.lower():
        model_cfg = AutoConfig.from_pretrained(Config.llava_usage_info)
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images
    else:
        raise NotImplementedError


def norm_projection(p_norm, trigger, eps):
    if p_norm == 'l2':
        norm = trigger.norm(p=2)
        if norm > eps:
            trigger = trigger * (eps / norm)
    elif p_norm == 'l_inf':
        torch.clamp(trigger, min=-eps, max=eps)
    else:
        raise NotImplementedError

    return trigger


def sliced_wasserstein_distance(features_A, features_B, num_projections=50):

    features_A = F.normalize(features_A, dim=-1)
    features_B = F.normalize(features_B, dim=-1)

    # choose a projection direction randomly
    projections = torch.randn(num_projections, features_A.size(1)).to(Config.device)
    projections = F.normalize(projections, dim=-1)
    
    # projection
    proj_features_A = features_A @ projections.T
    proj_features_B = features_B @ projections.T
    
    # Wasserstein distance
    w_distances = []
    for i in range(num_projections):
        proj_A = proj_features_A[:, i].sort()[0]
        proj_B = proj_features_B[:, i].sort()[0]
        w_distance = torch.mean((proj_A - proj_B) ** 2)
        w_distances.append(w_distance)
    
    return torch.mean(torch.tensor(w_distances))


def contrastive(ori_features, troj_features, temperature=0.1):

    ori_features = F.normalize(ori_features, dim=-1)  # [batch_size, feature_dim]
    troj_features = F.normalize(troj_features, dim=-1)  # [batch_size, feature_dim]

    N = ori_features.size(0)

    positive_pairs = []
    for i in range(N):
        non_corresponding_indices = torch.arange(N)
        non_corresponding_indices = non_corresponding_indices[non_corresponding_indices != i]
        random_positive_idx = non_corresponding_indices[torch.randint(0, len(non_corresponding_indices), (1,))]
        positive_pairs.append(random_positive_idx.item())

    positive_pairs = torch.tensor(positive_pairs)

    pos_similarity = torch.sum(troj_features * ori_features[positive_pairs], dim=-1) / temperature
    neg_similarity = torch.sum(troj_features * ori_features, dim=-1) / temperature

    loss = -torch.mean(pos_similarity) + torch.mean(neg_similarity)

    return loss


def utility(raw_features, clean_troj_features):
    raw_features = F.normalize(raw_features, dim=-1)
    clean_troj_features = F.normalize(clean_troj_features, dim=-1)

    utility_loss = - torch.sum(raw_features * clean_troj_features, dim=-1).mean() # utility loss

    return utility_loss


def compute_self_cos_sim(mat_a):
    # assert(mat_a.shape == mat_b.shape)
    # compute cosine similarity within one batch
    ele_size = mat_a.shape[0]
    mat_a = F.normalize(mat_a, dim=-1)
    sim_matrix = torch.mm(mat_a, mat_a.t())
    assert sim_matrix.shape == (ele_size, ele_size)

    sim_mask = (torch.ones_like(sim_matrix) - \
                torch.eye(ele_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(sim_mask).view(ele_size, -1)
    assert sim_matrix.shape == (ele_size, ele_size - 1)

    cos_sim = torch.mean(sim_matrix)
    return cos_sim


def target_loss(img_clean, trigger, clean_encoder, backdoored_encoder, target_feature, mean, std, patch_slice=None):
    if patch_slice is None:
        img_clean = img_clean.to(Config.device)
        img_triggered = torch.clamp(img_clean + trigger, 0, 1)
        img_triggered = (img_triggered - mean) / std
        img_clean = (img_clean - mean) / std
    else:
        img_clean = img_clean.to(Config.device)
        img_triggered = img_clean.clone().detach()
        img_triggered[patch_slice] = trigger
        img_triggered = (img_triggered - mean) / std
        img_clean = (img_clean - mean) / std

    raw_features = clean_encoder(img_clean)
    clean_troj_features = backdoored_encoder(img_clean)
    troj_troj_features = backdoored_encoder(img_triggered)

    effeciency_loss = - F.cosine_similarity(target_feature, troj_troj_features, dim=-1).mean() # backdoor effeciency loss
    utility_loss = - F.cosine_similarity(raw_features, clean_troj_features, dim=-1).mean() # utility loss

    return effeciency_loss, utility_loss


def PGD_cos(noise, backdoored_encoder, img_clean, mean, std, epsilon, alpha=2/255, num_steps=10):
    if not Config.Universe_PGD:
        noise = torch.rand_like(noise).to(Config.device)

    noise.requires_grad = True
    img_clean = img_clean.to(Config.device)

    for step in range(num_steps):

        img_noisy = torch.clamp(img_clean + noise, 0, 1)
        norm_img_noisy = (img_noisy - mean) / std
        
        noisy_features = backdoored_encoder(norm_img_noisy)

        loss_cos = - compute_self_cos_sim(noisy_features)
        loss_cos.backward()
        noise_grad = noise.grad.data
        noise = noise + alpha * noise_grad.sign()
        noise = torch.clamp(noise, -epsilon, epsilon)
        noise = noise.detach()  # clear the grad

        noise.requires_grad = True

    return noise


def PGD_reg(noise, trigger, backdoored_encoder, img_clean, mean, std, epsilon, alpha=2/255, num_steps=10, lambda_reg=1.0):
    if not Config.Universe_PGD:
        noise = torch.rand_like(noise).to(Config.device)
    
    noise.requires_grad = True
    img_clean = img_clean.to(Config.device)

    for step in range(num_steps):

        img_noisy = torch.clamp(img_clean + noise, 0, 1)
        norm_img_noisy = (img_noisy - mean) / std
        
        noisy_features = backdoored_encoder(norm_img_noisy)

        loss_cos = - compute_self_cos_sim(noisy_features)
        reg_noise_trigger = torch.norm(noise - trigger, p=2)
        loss_total = loss_cos - lambda_reg * reg_noise_trigger

        loss_total.backward()
        noise_grad = noise.grad.data
        noise = noise + alpha * noise_grad.sign()
        noise = torch.clamp(noise, -epsilon, epsilon)
        noise = noise.detach()  # clear the grad

        noise.requires_grad = True

    return noise


def PGD_ort(noise, trigger, backdoored_encoder, img_clean, mean, std, epsilon, alpha=2/255, num_steps=10, lambda_ort=1.0):
    if not Config.Universe_PGD:
        noise = torch.rand_like(noise).to(Config.device)
        print('not universal')
    
    noise.requires_grad = True
    img_clean = img_clean.to(Config.device)
    
    for step in range(num_steps):
        img_noisy = torch.clamp(img_clean + noise, 0, 1)
        norm_img_noisy = (img_noisy - mean) / std
        
        noisy_features = backdoored_encoder(norm_img_noisy)
        
        loss_cos = - compute_self_cos_sim(noisy_features)
        
        noise_flat = noise.view(noise.size(0), -1)
        trigger_flat = trigger.view(trigger.size(0), -1)
        
        loss_ort = F.cosine_similarity(noise_flat, trigger_flat, dim=-1).mean()
        
        total_loss = loss_cos + lambda_ort * loss_ort

        total_loss.backward()
        noise_grad = noise.grad.data
        noise = noise + alpha * noise_grad.sign()

        noise = torch.clamp(noise, -epsilon, epsilon)

        noise = noise.detach()
        noise.requires_grad = True

    return noise


def random_noise(trigger, epsilon):
    
    noise = torch.rand_like(trigger) * 2 * epsilon - epsilon

    return noise