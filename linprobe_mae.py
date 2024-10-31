import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets
from torchvision.transforms import v2
import numpy as np
import argparse
import wandb
import os
import math
import sys
from pathlib import Path

from segmentation_pipeline import get_dataset as get_segmentation_dataset
from classification_pipeline import get_dataset as get_classification_dataset

from mae.models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from mae.utils import add_weight_decay
from mae.utils import NativeScalerWithGradNormCount as NativeScaler
from mae.utils import save_model, load_model, adjust_learning_rate, SmoothedValue
from mae.pos_embed import interpolate_pos_embed
from mae.lars import LARS
from timm.models.layers import trunc_normal_

from PIL import Image

from classification_pipeline import evaluate


def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """

    if args.dataset == 'cub':
        num_classes = 200
    elif args.dataset == 'fmnist':
        num_classes = 10
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #switch to single channel
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'celeba':
        num_classes = 10177
    elif args.dataset == 'caltech101':
        num_classes = 101
    else:
        raise ValueError()

    if args.model == 'vit_base_patch16':
        model = vit_base_patch16(num_classes = num_classes, global_pool = args.global_pool)
    elif args.model == 'vit_large_patch16':
        model = vit_large_patch16(num_classes = num_classes, global_pool = args.global_pool)
    elif args.model == 'vit_huge_patch14':
        model = vit_huge_patch14(num_classes = num_classes, global_pool = args.global_pool)
    else:
        raise ValueError()
    
    return model

def train_one_epoch(model, W, criterion, dataloader, optimizer, device, epoch, loss_scaler, args=None):
    model.train()
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    running_loss = 0.0
    running_corrects = 0
    for data_iter_step, (images, targets) in enumerate(dataloader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(dataloader) + epoch, args)

        images = images.to(device)
        targets = targets.to(device)

        if W:
            flattened = images.flatten(1)
            down_projected = flattened @ W.weight.T
            reconstructed = down_projected @ W.weight
            images = reconstructed.reshape(images.shape)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        running_loss += loss_value
        running_corrects += torch.sum(preds == targets.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double().item() / len(dataloader.dataset)

    return {'loss': epoch_loss, 'acc': epoch_acc}



def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation dataset pipeline")

    #CUB dataset-related parameters
    parser.add_argument('--certainty_threshold', type=int, default=4, help='Threshold for certainty levels in CUBS dataset.')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None, help='Directory where the dataset is located.')
    parser.add_argument('--dataset', type=str, choices=['cub', 'celebamask'], required=True, help="Dataset being evaluated on.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")
    
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--dataset_task', type=str, choices=['classification', 'segmentation'], help='Choose a downstream task to pull the dataset from.')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    
    parser.add_argument('--finetune', type=str, default='', help='finetune from checkpoint')
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--model', type=str, 
                        choices=['vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14'],
                        default='vit_base_patch16', 
                        help="Backbone model being evaluated on.")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')

    # Projection parameters
    parser.add_argument('--k', type=int, help="Dimension of latent space from projection")
    parser.add_argument('--projection', action='store_true', help="Flag that turns on projection bottleneck W")
    parser.add_argument('--w_weights', type=str, default='', help='Path to load the weights of a pretrained projection W.')

    # Logging parameters
    parser.add_argument('--experiment', type=str, default='default', help="Experiment name, use for grouping in wandb")
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Useful for slurm purposes.")
    parser.add_argument('--no_log', action='store_true', help="Flag that turns off wandb. Useful during debugging.")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    transform_train = v2.Compose([
            v2.RandomResizedCrop((224, 224), interpolation=3),  # 3 is bicubic
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = v2.Compose([
        v2.Resize(256, interpolation=3),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # I was right, linear probe MAE always needs to do classification at the end of the day.
    train_dataset, test_dataset = get_classification_dataset(args, transform_train=transform_train, transform_val=transform_val)
    
    #TODO: I think this and the pretrain_mae.py version will end up being useless
    # if args.dataset_task == 'classification':
        # train_dataset, test_dataset = get_classification_dataset(args, transform_train=transform_train, transform_val=transform_val)
    # else:
    #     train_dataset, test_dataset, classes = get_segmentation_dataset(args, transform_train=transform_train, transform_val=transform_val)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print(f'Load pre-trained checkpoint from: {args.finetune}')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model = model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * 1
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='mean') #in this case, I think it's okay to include the background
    if args.projection:
        W = torch.nn.Linear(train_dataset[0][0].flatten().shape[0], args.k, bias=False, device=device)
        W.load_state_dict(torch.load(args.w_weights))
        I = torch.eye(args.k, device=device)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()
    criterion = nn.CrossEntropyLoss()

    load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    saved_checkpoints = []
    
    if not args.no_log:
        wandb.init(
            project="mae_linprobe",
            name=f'{args.experiment}-{args.dataset}-{args.dataset_task}-{args.trial}',
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "architecture": args.model,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "K": args.k,
                "projection": args.projection,
                "experiment": args.experiment,
                "trial": args.trial,
            },
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
    print(args)

    best_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'epoch: {epoch}')
        if args.projection:
            train_stats = train_one_epoch(model, W, criterion, train_loader, optimizer, device, epoch, loss_scaler, args=args)
            val_stats = evaluate(args, model, W, criterion, test_loader, device=device)
        else:
            train_stats = train_one_epoch(model, None, criterion, train_loader, optimizer, device, epoch, loss_scaler, args=args)
            val_stats = evaluate(args, model, None, criterion, test_loader, device=device)
    
        if not args.no_log:
            log = {'epoch': epoch,}
            for k,v in train_stats.items():
                log.update({f'train/{k}': v})
            for k,v in val_stats.items():
                log.update({f'val/{k}': v})

            if args.projection:
                log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum()
            wandb.log(log)

        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            best_epoch = epoch
            os.makedirs(args.output, exist_ok=True)
            save_model(args, epoch, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

            output_dir = Path(args.output)
            epoch_name = str(epoch)
            # checkpoint_path = output_dir / (f'{args.experiment}-{args.dataset}-{args.k}-checkpoint-{epoch_name}.pth')
            checkpoint_path = output_dir / (f'{args.experiment}-{args.dataset}-{args.k}-checkpoint.pth')
            # Append the saved checkpoint to the list
            saved_checkpoints.append(checkpoint_path)
            if len(saved_checkpoints) > 1:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)


            
        print(train_stats)
        print(val_stats)

    print(f'Best loss: {best_acc:4f} at epoch {best_epoch}')

    if not args.no_log:
        wandb.finish()
