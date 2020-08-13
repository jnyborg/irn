import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import l8biome.dataloader
import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader, criterion):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss1', 'acc')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = criterion(x, label)
            acc = accuracy(x, label)

            val_loss_meter.add({'loss1': loss1.item(), 'acc': acc})

    model.train()

    val_loss = val_loss_meter.pop('loss1')
    val_acc = val_loss_meter.pop('acc')
    print('loss: %.4f' % val_loss, 'acc: %.4f' % val_acc)

    return val_loss


def accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        preds = outputs.argmax(1)
        return preds.eq(targets).float().mean().item()


def run(args):
    if args.dataset == 'l8biome':
        model = getattr(importlib.import_module(args.cam_network), 'Net')(n_classes=2, in_channels=10, pretrained=False)
        train_dataset = l8biome.dataloader.L8BiomeDataset(args.data_root, 'train')
        val_dataset = l8biome.dataloader.L8BiomeDataset(args.data_root, 'val')
        criterion = F.cross_entropy  # clear vs cloudy
    else:
        model = getattr(importlib.import_module(args.cam_network), 'Net')(n_classes=20, in_channels=3)
        train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.data_root,
                                                                    resize_long=(320, 640), hor_flip=True,
                                                                    crop_size=512, crop_method="random")

        val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.data_root,
                                                                  crop_size=512)
        criterion = F.multilabel_soft_margin_loss

    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    print(model)
    print(f"Number of parameters: {sum([p.numel() for p in model.parameters()]):,}")

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss1', 'acc')

    timer = pyutils.Timer()
    best_val_loss = np.inf

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))

        for step, pack in enumerate(tqdm(train_data_loader, f'Epoch {ep + 1}/{args.cam_num_epoches}')):

            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss = criterion(x, label)

            avg_meter.add({'loss1': loss.item(), 'acc': accuracy(x, label)})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'acc:%.4f' % (avg_meter.pop('acc')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            val_loss = validate(model, val_data_loader, criterion)
            if val_loss < best_val_loss:
                print(f'Validation loss improved from {best_val_loss} to {val_loss}, saving model')
                torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
                best_val_loss = val_loss
            else:
                print(f'Validation loss did not improve from {best_val_loss}')

            timer.reset_stage()

    torch.cuda.empty_cache()
