import os
import math
import torch
import random
import numpy as np
import blobfile as bf

from PIL import Image
from pathlib import Path

from utils.register import Registers
from torch.utils.data import Dataset
from .utils import preprocess_dataset_config, _list_image_files_recursively


@Registers.datasets.register_with_name('custom_semantic_synthesis')
class CustomSemanticSynthesisDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.dataset_config = preprocess_dataset_config(dataset_config, stage)

        dataset_name = self.dataset_config.dataset_name
        dataset_path = self.dataset_config.dataset_path

        if dataset_name == 'ADE20K':
            all_files = _list_image_files_recursively(os.path.join(dataset_path, f'{stage}/A'))  # 'images', stage))
            classes = _list_image_files_recursively(os.path.join(dataset_path, f'{stage}/B'))  # 'annotations', stage))
            instances = None
        else:
            raise NotImplementedError

        self.dataset = ImageDataset(
            dataset_mode=dataset_name,
            resolution=dataset_config.image_size,
            image_paths=all_files,
            classes=classes,
            instances=instances,
            shard=0,
            num_shards=1,
            random_crop=self.dataset_config.random_crop,
            random_flip=self.dataset_config.flip,
            is_train=(stage == 'train')
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class ImageDataset(Dataset):
    def __init__(self,
                 dataset_mode,
                 resolution,
                 image_paths,
                 classes=None,
                 instances=None,
                 shard=0,
                 num_shards=1,
                 random_crop=False,
                 random_flip=True,
                 is_train=True):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, 'rb') as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        image_name = Path(path).stem

        out_dict = {}
        class_path = self.local_classes[idx]
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[idx]  # DEBUG: from classes to instances, may affect CelebA
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution)
        else:
            if self.is_train:
                if self.random_crop:
                    arr_image, arr_class, arr_instance = random_crop_arr([pil_image, pil_class, pil_instance],
                                                                         self.resolution)
                else:
                    arr_image, arr_class, arr_instance = center_crop_arr([pil_image, pil_class, pil_instance],
                                                                         self.resolution)
            else:
                arr_image, arr_class, arr_instance = resize_arr([pil_image, pil_class, pil_instance], self.resolution,
                                                                keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None

        arr_image = arr_image.astype(np.float32) / 127.5 - 1

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()

        if self.dataset_mode == 'ADE20K':
            arr_class = arr_class - 1
            arr_class[arr_class == 255] = 150
        elif self.dataset_mode == 'coco':
            arr_class[arr_class == 255] = 182

        # out_dict['label'] = arr_class[None,]
        # if arr_instance is not None:
        #     out_dict['instance'] = arr_instance[None,]
        
        arr_class = arr_class[None,]
        image = np.transpose(arr_image, [2, 0, 1])
        c, h, w = image.shape
        
        image = torch.FloatTensor(image)
        image_cond = torch.cat((torch.tensor(arr_class), torch.tensor(arr_class), torch.tensor(arr_class)), dim=0)
        image_cond = image_cond / 127.5 - 1.
        one_hot_index = torch.tensor(arr_class, dtype=torch.int64)
        one_hot_label = torch.FloatTensor(151, h, w).zero_()
        one_hot_label = one_hot_label.scatter_(0, one_hot_index, 1.0)

        return (image, image_name), (image_cond, image_name), one_hot_label


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    return arr_image, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size], \
        arr_instance[crop_y: crop_y + image_size, crop_x: crop_x + image_size] if arr_instance is not None else None
