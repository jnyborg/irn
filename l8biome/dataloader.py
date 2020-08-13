import albumentations as alb
import cv2
from tifffile import tifffile
from torch.utils import data
import torch
import os
import random
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

import voc12.dataloader
from misc import imutils


class L8BiomeDataset(data.Dataset):
    def __init__(self, root, mode='train', mask_file='mask.tif', transform=None):
        self.root = root = os.path.join(root, mode)
        classes, class_to_idx = self._find_classes(root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = self._make_dataset(root, class_to_idx)
        self.num_channels = 10

        if transform is None:
            transform = []
            if mode == 'train':
                transform.append(alb.HorizontalFlip())
            transform.append(
                alb.Normalize(mean=(0.5,) * self.num_channels, std=(0.5,) * self.num_channels, max_pixel_value=2 ** 16 - 1))
            transform.append(ToTensorV2())
            self.transform = alb.Compose(transform)
        else:
            self.transform = transform

        self.is_segmentation = mask_file is not None
        self.mask_file = mask_file

    def __getitem__(self, index):
        patch_dir, label, name = self.images[index]
        image = tifffile.imread(os.path.join(patch_dir, 'image.tif'))

        out = {'name': name, 'label': torch.tensor(label).long()}
        if self.is_segmentation:
            mask = self.load_mask(patch_dir)
            transformed = self.transform(image=image, mask=mask)
            out['img'] = transformed['image']
            out['mask'] = transformed['mask']
        else:
            out['img'] = self.transform(image=image)['image']
        return out

    def load_mask(self, patch_dir):
        # 0 = invalid, 1 = clear, 2 = clouds
        return tifffile.imread(os.path.join(patch_dir, self.mask_file)).astype(np.long)

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

                label = self.class_to_idx[target]

                # convert to onehot for compatability
                # onehot = np.zeros(len(self.classes))
                # onehot[label] = 1
                images.append((patch_dir, label, self._make_name(patch_dir)))

        return images

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class L8BiomeDatasetMSF(L8BiomeDataset):
    def __init__(self, root, mode='train', mask_file='mask.tif', scales=(1.0,)):
        self.scales = scales

        super().__init__(root, mode, mask_file, transform=alb.NoOp())
        self.scales = scales

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        ms_img_list = []
        img = sample['img']
        for s in self.scales:
            height, width = img.shape[:2]
            target_height, target_width = (int(np.round(height * s)), int(np.round(width * s)))

            transform = []
            if s != 1:
                transform.append(alb.Resize(target_height, target_width, interpolation=cv2.INTER_CUBIC))
            transform.append(alb.Normalize((0.5,)*self.num_channels, (0.5,)*self.num_channels, 2**16-1))
            transform = alb.Compose(transform)
            s_img = transform(image=img)['image']
            s_img = imutils.HWC_to_CHW(s_img)

            # ResNet50 CAM compute cam as sum of original + flipped image flipped back
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": sample['name'], "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": sample['label']}
        return out


if __name__ == '__main__':
    l8biome = L8BiomeDataset('/home/jnyborg/git/fixed-point-gan/data/L8Biome')
    print('voc12')
    voc12 = voc12.dataloader.VOC12ClassificationDataset("voc12/train_aug.txt", 'data/VOC2012',
                                       resize_long=(320, 640), hor_flip=True,
                                       crop_size=512, crop_method="random")


    print('l8biome')
    sample = l8biome[0]
    img, label = sample['img'], sample['label']
    print(type(img), img.shape, img.dtype)
    print(type(label), label.shape, label.dtype)
    print(label)
    #
    # print('voc12')
    # sample = voc12[0]
    # img, label = sample['img'], sample['label']
    # print(type(img), img.shape, img.dtype)
    # print(type(label), label.shape, label.dtype)
    # print(label)

    # MSF
    l8biome = L8BiomeDatasetMSF('/home/jnyborg/git/fixed-point-gan/data/L8Biome', scales=(1.0, 0.5, 1.5, 2.0))

    sample = l8biome[0]
    print(sample['size'])
    print([i.shape for i in sample['img']])


    voc12 = voc12.dataloader.VOC12ClassificationDatasetMSF("voc12/train_aug.txt", 'data/VOC2012',
                                                           scales=(1.0, 0.5, 1.5, 2.0))
    sample = voc12[0]
    print(sample['size'])
    print([i.shape for i in sample['img']])
