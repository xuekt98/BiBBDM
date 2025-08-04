from PIL import Image
from pathlib import Path
from random import random, randint
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_paths_cond=None, **dataset_config):
        self.image_paths = image_paths
        self.image_paths_cond = image_paths_cond
        self._length = len(image_paths)

        self.flip = dataset_config['flip']
        self.to_normal = dataset_config['to_normal']  # normalize to [-1, 1]
        self.image_size = dataset_config['image_size']
        self.resize = dataset_config['resize']
        self.random_crop = dataset_config['random_crop']
        self.crop_p1 = dataset_config['crop_p1']
        self.crop_p2 = dataset_config['crop_p2']

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        def read_image(image_path, cond_image_path=None):
            image, image_cond = None, None
            has_cond = True if cond_image_path is not None else False
            try:
                image = Image.open(image_path)
                image_cond = Image.open(cond_image_path) if has_cond else None
            except BaseException as e:
                print(image_path)

            if not image.mode == 'RGB':
                image = image.convert('RGB')
                image_cond = image_cond.convert('RGB') if has_cond else None
            width_image, height_image = image.size
            width_cond, height_cond = image_cond.size if has_cond else image.size
            height_resize, width_resize = self.image_size
            if self.resize:
                if self.random_crop:
                    crop_type = random()
                    if crop_type < self.crop_p1:
                        transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.RandomHorizontalFlip(p=p),
                            transforms.ToTensor()
                        ])
                        image = transform(image)
                        image_cond = transform(image_cond) if has_cond else None
                    else:
                        if width_image < width_resize or height_image < height_resize \
                                or width_image != width_cond or height_image != height_cond \
                                or crop_type < self.crop_p2:
                            h, w = self.image_size
                            transform = transforms.Compose([
                                transforms.Resize((h * 2, w * 2)),
                                transforms.RandomHorizontalFlip(p=p),
                                transforms.ToTensor()
                            ])
                            image = transform(image)
                            image_cond = transform(image_cond) if has_cond else None
                        else:
                            transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=p),
                                transforms.ToTensor()
                            ])
                            image = transform(image)
                            image_cond = transform(image_cond) if has_cond else None

                        c, h, w = image.shape
                        h_r, w_r = self.image_size
                        rand_w, rand_h = randint(0, w - w_r), randint(0, h - h_r)
                        image = image[:, rand_h:rand_h + h_r, rand_w:rand_w + w_r]
                        if has_cond:
                            image_cond = image_cond[:, rand_h:rand_h + h_r, rand_w:rand_w + w_r]
                else:
                    transform = transforms.Compose([
                        transforms.Resize(self.image_size),
                        transforms.RandomHorizontalFlip(p=p),
                        transforms.ToTensor()
                    ])
                    image = transform(image)
                    image_cond = transform(image_cond) if has_cond else None
            else:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=p),
                    transforms.ToTensor()
                ])
                image = transform(image)
                image_cond = transform(image_cond) if has_cond else None

            if self.to_normal:
                image = (image - 0.5) * 2.
                image.clamp_(-1., 1.)
            if has_cond:
                image_cond = (image_cond - 0.5) * 2.
                image_cond.clamp_(-1., 1.)
            return image, image_cond

        img_path = self.image_paths[index]
        image_name = Path(img_path).stem

        if self.image_paths_cond is not None:
            cond_img_path = self.image_paths_cond[index]
            cond_image_name = Path(cond_img_path).stem
            image, cond_image = read_image(img_path, cond_img_path)

            return (image, image_name), (cond_image, cond_image_name)
        image, _ = read_image(img_path)
        return image, image_name
