import torch, torchvision
import PIL
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset

_dataset_name = ['llava', 'minigpt']

_mean = {
    'llava': [0.485, 0.456, 0.406],
    'minigpt': [0.485, 0.456, 0.406]
}

_std = {
    'llava': [0.229, 0.224, 0.225],
    'minigpt': [0.229, 0.224, 0.225]
}

_size = {
    'llava': (336, 336),
    'minigpt': (448, 448)
}


imagenet_path = './shadow_dataset/imagenet'

class ImageNetTensorDataset(Dataset):
    def __init__(self, dataset, transform):
        assert isinstance(dataset, Dataset)
        self.targets = dataset.targets
        self.filename = [t[0] for t in dataset.imgs]
        self.classes = dataset.classes
        self.transform = transform
        assert self.transform is not None

    def __getitem__(self, index):
        img = PIL.Image.open(self.filename[index]).convert('RGB')
        img = self.transform(img) # [0,1] tensor (C,H,W)
        img_tensor = img.clone().to(dtype=torch.float64)
        img_tensor = (img_tensor.permute(1,2,0) * 255).type(torch.uint8) # [0, 255] tensor (H,W,C)
        return img_tensor, self.targets[index]

    def __len__(self):
        return len(self.targets)
    
    def rand_sample(self, ratio):
        idx = random.sample(range(len(self.targets)),
                            int(len(self.targets) * ratio))
        self.targets = [ self.targets[i] for i in idx]
        self.filename =[ self.filename[j] for j in idx]

def getTensorImageNet(transform, split='val'):
    assert(split in ['val', 'train'])
    imagenet_dataset = torchvision.datasets.ImageNet(
            imagenet_path,
            split=split, transform=None)

    tensor_imagenet = ImageNetTensorDataset(imagenet_dataset, transform)
    return tensor_imagenet

def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize

def get_resize(size):
    if isinstance(size, str):
        assert size in _dataset_name, _dataset_name
        size = _size[size]
    return transforms.Resize(size)

def get_processing(dataset, size, augment=True, is_tensor=False, need_norm=True, ):

    transforms_list = []
    detransforms_list = []
    transforms_list.append(get_resize(size))
    if augment:
        transforms_list.append(transforms.RandomResizedCrop(_size[dataset], scale=(0.2, 1.)))
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.CenterCrop(_size[dataset]))
    
    if not is_tensor:
        transforms_list.append(transforms.ToTensor())
    if need_norm:
        normalize, unnormalize = get_norm(dataset)
        transforms_list.append(normalize)
        detransforms_list.append(unnormalize)
        

    preprocess = transforms.Compose(transforms_list)
    deprocess  = transforms.Compose(detransforms_list)
    
    return preprocess, deprocess
