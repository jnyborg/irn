from albumentations import HorizontalFlip, Normalize, Compose
from tifffile import tifffile
from torch.utils import data
import torch
import os
import random
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from voc12.dataloader import VOC12ClassificationDataset


class L8BiomeDataset(data.Dataset):
    def __init__(self, root, mode='train', mask_file='mask.tif', keep_ratio=1.0):
        self.root = root = os.path.join(root, mode)
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = self._make_dataset(root, class_to_idx)
        if keep_ratio < 1.0:
            # Subsample images for supervised training on fake images, and fine-tuning on keep_ratio% real images
            print('Dataset size before keep_ratio', len(self.images))
            random.seed(42)  # Ensure we pick the same 1% across experiments
            random.shuffle(self.images)
            self.images = self.images[:int(keep_ratio * len(self.images))]
            print('Dataset size after keep_ratio', len(self.images))

        self.num_channels = 10
        transform = []
        if mode == 'train':
            transform.append(HorizontalFlip())
        transform.append(Normalize(mean=(0.5,) * self.num_channels, std=(0.5,) * self.num_channels, max_pixel_value=2 ** 16 - 1))
        transform.append(ToTensorV2())
        self.transform = Compose(transform)

        self.is_segmentation = mask_file is not None
        self.mask_file = mask_file

    def __getitem__(self, index):
        patch_dir, label, name = self.images[index]
        image = tifffile.imread(os.path.join(patch_dir, 'image.tif'))

        out = {'name': name, 'label': torch.tensor(label).float()}
        if self.is_segmentation:
            # 0 = invalid, 1 = clear, 2 = clouds
            mask = tifffile.imread(os.path.join(patch_dir, self.mask_file)).astype(np.long)
            transformed = self.transform(image=image, mask=mask)
            out['img'] = transformed['image']
            out['mask'] = transformed['mask']
        else:
            out['img'] = self.transform(image=image)['image']
        return out


    def __len__(self):
        return len(self.images)

    def _make_name(self, path):
        return '_'.join(path.split('/')[-3:])

    def _make_dataset(self, root, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for patch_dir, _, file_names in sorted(os.walk(d)):
                if len(file_names) == 0:
                    continue

                # convert to onehot for compatability
                label = self.class_to_idx[target]
                onehot = np.zeros(len(self.classes))
                onehot[label] = 1
                images.append((patch_dir, onehot, self._make_name(patch_dir)))

        return images

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


if __name__ == '__main__':
    l8biome = L8BiomeDataset('/home/jnyborg/git/fixed-point-gan/data/L8Biome')
    print('voc12')
    voc12 = VOC12ClassificationDataset("voc12/train_aug.txt", 'data/VOC2012',
                                       resize_long=(320, 640), hor_flip=True,
                                       crop_size=512, crop_method="random")

    print('l8biome')
    sample = l8biome[0]
    img, label = sample['img'], sample['label']
    print(type(img), img.shape, img.dtype)
    print(type(label), label.shape, label.dtype)
    print(label)

    print('voc12')
    sample = voc12[0]
    img, label = sample['img'], sample['label']
    print(type(img), img.shape, img.dtype)
    print(type(label), label.shape, label.dtype)
    print(label)


