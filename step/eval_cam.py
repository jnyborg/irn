
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from tqdm import tqdm

import l8biome.dataloader

def run(args):
    if args.dataset == 'l8biome':
        dataset = l8biome.dataloader.L8BiomeDataset(args.data_root, 'train', mask_file='mask.tif')
        # Only compute CAM for cloudy images - we know the segmentation label for clear already.
        dataset.images = [img for img in dataset.images if 'cloudy' in img[2]]
        labels = [dataset.load_mask(x[0]) for x in dataset.images]
        ids = [x[2] for x in dataset.images]
    else:
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.data_root)
        labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
        ids = dataset.ids

    preds = []
    for id in tqdm(ids):
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        if args.dataset == 'l8biome':
            # background class (in our case 'clear') corresponds to class 0 already
            keys = np.pad(cam_dict['keys'], (1, 0), mode='constant')
        else:
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

    if args.dataset == 'l8biome':
        # Compute metrics as FCD
        pass
    else:
        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        print({'iou': iou, 'miou': np.nanmean(iou)})
