import os
import torch
import random
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from utils.register import Registers
from torch.utils.data import Dataset
from .utils import get_image_paths_from_dir, preprocess_dataset_config, read_image, transform_image


class ImagePathDataset(Dataset):
    def __init__(self, dataset_config, image_paths, image_paths_cond=None):
        self.image_paths = image_paths
        self.image_paths_cond = image_paths_cond
        self._length = len(image_paths)

        self.dataset_config = dataset_config
        self.flip = self.dataset_config.flip

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        flip = False
        if index >= self._length:
            index = index - self._length
            flip = True

        img_path = self.image_paths[index]
        image, image_name = read_image(img_path)

        if self.image_paths_cond is not None:
            cond_img_path = self.image_paths_cond[index]
            cond_image, cond_image_name = read_image(cond_img_path)

            image, cond_image = transform_image(self.dataset_config, image, cond_image, flip=flip)
            return (image, image_name), (cond_image, cond_image_name)
        image, _ = transform_image(self.dataset_config, image, flip=flip)
        return image, image_name


@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.dataset_config = preprocess_dataset_config(dataset_config, stage)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.imgs = ImagePathDataset(self.dataset_config, image_paths)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.dataset_config = preprocess_dataset_config(dataset_config, stage)
        if self.dataset_config.direction == 'BtoA':
            image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
            image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        elif self.dataset_config.direction == 'AtoB':
            image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
            image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.imgs = ImagePathDataset(self.dataset_config, image_paths_ori, image_paths_cond)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i]


@Registers.datasets.register_with_name('custom_unpaired')
class CustomUnpairedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.dataset_config = preprocess_dataset_config(dataset_config, stage)
        if self.dataset_config.direction == 'BtoA':
            image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
            image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        elif self.dataset_config.direction == 'AtoB':
            image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
            image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.imgs_ori = ImagePathDataset(self.dataset_config, image_paths_ori)
        self.imgs_cond = ImagePathDataset(self.dataset_config, image_paths_cond)
    
    def __len__(self):
        return len(self.imgs_ori)
    
    def __getitem__(self, index):
        ori_index = index
        cond_index = random.randint(0, len(self.imgs_cond) - 1)

        return self.imgs_ori[ori_index], self.imgs_cond[cond_index]
