import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
from tqdm import tqdm

import voc12.dataloader
import l8biome.dataloader
from misc import torchutils, imutils
import sys

cudnn.enabled = True

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            # Run through each scale of image
            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']]

            # Each output is resized to strided_size (lower than original) and summed
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            # Each output is resized to strided_up_size (which should be orignal size?)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            # Pick the cams corresponding to image-level labels
            # Normalize by max value across H x W dimension for each channel
            valid_cat = torch.nonzero(label, as_tuple=False)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')
                sys.stdout.flush()


def run(args):

    n_gpus = torch.cuda.device_count()

    if args.dataset == 'l8biome':
        model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=2, in_channels=10, pretrained=False)
        dataset = l8biome.dataloader.L8BiomeDatasetMSF(args.data_root, 'train', scales=args.cam_scales)
        # Only compute for cloudy images, clear should have empty mask
        dataset.images = [img for img in dataset.images if 'cloudy' in img[2]]
    else:
        model = getattr(importlib.import_module(args.cam_network), 'CAM')(n_classes=20)
        dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                                 voc12_root=args.data_root, scales=args.cam_scales)

    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()