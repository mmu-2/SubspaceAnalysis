from cv2 import transform
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

from mae.models_mae import mae_vit_mu_patch2, mae_vit_tiny_patch16, mae_vit_tiny_patch2, \
    mae_vit_small_patch16, mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from mae.utils import add_weight_decay
from mae.utils import NativeScalerWithGradNormCount as NativeScaler
from mae.utils import save_model, load_model, adjust_learning_rate, SmoothedValue

from PIL import Image



def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """

    if args.dataset == 'cub': 
        img_size = 224
        in_chans = 3
    elif args.dataset == 'fmnist': 
        img_size = 28
        in_chans = 1
    elif args.dataset == 'cifar10':
        img_size = 32
        in_chans = 3
    elif args.dataset == 'celeba': 
        img_size = 224
        in_chans = 3
    elif args.dataset == 'caltech101':
        img_size = 224
        in_chans = 3
    else:
        raise ValueError()

    if args.model == 'mae_vit_mu_patch2':
        model = mae_vit_mu_patch2(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_tiny_patch16':
        model = mae_vit_tiny_patch16(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_tiny_patch2':
        model = mae_vit_tiny_patch2(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_small_patch16':
        model = mae_vit_small_patch16(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_base_patch16':
        model = mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_large_patch16':
        model = mae_vit_large_patch16(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    elif args.model == 'mae_vit_huge_patch14':
        model = mae_vit_huge_patch14(norm_pix_loss=args.norm_pix_loss, img_size=img_size, in_chans=in_chans)
    else:
        raise ValueError()
    
    return model

def to_image(image_tensor: torch.Tensor):
    """Processes a pytorch tensor into PIL format."""
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    # image is [H, W, 3], we normalize and convert to uint8
    # print('image_tensor.shape',image_tensor.shape)
    # print('image_tensor.dtype',image_tensor.dtype)
    # print('imagenet_std.shape',imagenet_std.shape)
    # print('imagenet_std.dtype',imagenet_std.dtype)
    # image_tensor = image_tensor.to(torch.float32)
    image_tensor = image_tensor * imagenet_std + imagenet_mean
    image_tensor = image_tensor * 255
    image = torch.clip(image_tensor, 0, 255).int().numpy()
    # convert to PIL image and save
    image_pil = Image.fromarray(image.astype(np.uint8))
    return image_pil

def train_one_epoch(model, W, dataloader, optimizer, device, epoch, loss_scaler, args=None):
    model.train()

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    running_loss = 0.0
    for data_iter_step, (images, target_masks) in enumerate(dataloader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(dataloader) + epoch, args)

        images = images.to(device)
        target_masks = target_masks.to(device)
        unprojected_image = images.detach().cpu()
        unprojected_image = torch.einsum('nchw->nhwc', unprojected_image)

        if W:
            flattened = images.flatten(1)
            down_projected = flattened @ W.weight.T
            reconstructed = down_projected @ W.weight
            images = reconstructed.reshape(images.shape)

        with torch.amp.autocast('cuda'):
            loss, y, mask = model(images, mask_ratio=args.mask_ratio)

        ### Visualization Code ###        
        x = images.detach().cpu()
        # assuming x, y, and mask are already defined
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.in_chans)  # (N, H*W, p*p*C)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        x = torch.einsum('nchw->nhwc', x)
        # masked image
        im_masked = x * (1 - mask)
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        
        if args.train_projection:
            loss = loss + (W.weight @ W.weight.T - I).square().sum()
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
    lr = optimizer.param_groups[0]["lr"]

    return {'running_loss': running_loss,
            'lr': lr,
            'unprojected_image': to_image(unprojected_image[0]),
            'image': to_image(x[0]),
            'masked': to_image(im_masked[0]),
            'reconstruction': to_image(y[0]),
            'reconstruction_plus_visible': to_image(im_paste[0]),
            }

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation dataset pipeline")

    #CUB dataset-related parameters
    parser.add_argument('--certainty_threshold', type=int, default=4, help='Threshold for certainty levels in CUBS dataset.')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=None, help='Directory where the dataset is located.')
    parser.add_argument('--dataset', type=str, choices=['cub', 'fmnist', 'celebamask', 'cifar10', 'caltech101'], required=True, help="Dataset being evaluated on.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")
    
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--dataset_task', type=str, choices=['classification', 'segmentation'], help='Tells me where to pull the dataset from.')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--data_aug', action='store_true', help="Flag that turns on data augmentations.")

    
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--model', type=str, 
                        choices=['mae_vit_mu_patch2','mae_vit_tiny_patch16','mae_vit_tiny_patch2',
                                 'mae_vit_small_patch16','mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'],
                        default='mae_vit_base_patch16', 
                        help="Backbone model being evaluated on.")
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Projection parameters
    parser.add_argument('--k', type=int, help="Dimension of latent space from projection")
    parser.add_argument('--projection', action='store_true', help="Flag that turns on projection bottleneck W")
    parser.add_argument('--train_projection', action='store_true', help="Flag that updates projection W weights. Projection flag must also be on for this.")
    parser.add_argument('--w_weights', type=str, default='', help='Path to load the weights of a pretrained projection W.')

    # Logging parameters
    parser.add_argument('--experiment', type=str, default='default', help="Experiment name, use for grouping in wandb")
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Useful for slurm purposes.")
    parser.add_argument('--no_log', action='store_true', help="Flag that turns off wandb. Useful during debugging.")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    if args.dataset == 'cub':
        transform_train = v2.Compose([
                v2.RandomResizedCrop((224, 224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform_train = None
    
    if args.dataset_task == 'classification':
        # transform_val doesn't matter at the moment because we don't use it.
        train_dataset, test_dataset = get_classification_dataset(args, transform_train=transform_train, transform_val=transform_train)
    else:
        train_dataset, test_dataset, classes = get_segmentation_dataset(args, transform_train=transform_train, transform_val=transform_train)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model = model.to(device)

    if args.projection:
        W = torch.nn.Linear(train_dataset[0][0].flatten().shape[0], args.k, bias=False, device=device)
        if not args.train_projection:
            W.load_state_dict(torch.load(args.w_weights, weights_only=True))
        I = torch.eye(args.k, device=device)


    param_groups = add_weight_decay(model, args.weight_decay)
    if args.train_projection:
        param_groups.append(
            {'params': W.parameters(), 'weight_decay': 0.} #TODO: consider weight decay in the future.
        )

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # Don't plan to train W at this time.
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()
    load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # List to store saved checkpoints
    saved_checkpoints = []
    
    if not args.no_log:
        wandb.init(
            project="mae_pretrain",
            name=f'{args.experiment}-{args.dataset}-{args.trial}',
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        print(args)

    best_loss = 999999
    best_epoch = 0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'epoch: {epoch}')
        if args.projection:
            train_stats = train_one_epoch(model, W, train_loader, optimizer, device, epoch, loss_scaler, args=args)
        else:
            train_stats = train_one_epoch(model, None, train_loader, optimizer, device, epoch, loss_scaler, args=args)

        unprojected_image = train_stats.pop('unprojected_image', None)
        image = train_stats.pop('image', None)
        masked = train_stats.pop('masked', None)
        reconstruction = train_stats.pop('reconstruction', None)
        reconstruction_plus_visible = train_stats.pop('reconstruction_plus_visible', None)
        

        if not args.no_log:
            log = {'epoch': epoch,}
            for k,v in train_stats.items():
                log.update({f'train/{k}': v})

            if epoch % 100 == 0:
                log.update({"unprojected_image": wandb.Image(unprojected_image)})
                log.update({"image": wandb.Image(image)})
                log.update({"masked": wandb.Image(masked)})
                log.update({"reconstruction": wandb.Image(reconstruction)})
                log.update({"reconstruction_plus_visible": wandb.Image(reconstruction_plus_visible)})

            if args.projection:
                log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum()

            wandb.log(log)
        

        if train_stats['running_loss'] < best_loss:
        # if (epoch+1) % 500 == 0:
            best_loss = train_stats['running_loss']
            best_epoch = epoch
            os.makedirs(args.output, exist_ok=True)
            output_dir = Path(args.output)

            # checkpoint_path = output_dir / (f'{args.experiment}-{args.dataset}-{args.trial}-{args.dataset_task}-pretrain-checkpoint{epoch}.pth')
            checkpoint_path = output_dir / (f'{args.experiment}-{args.dataset}-{args.k}-{args.dataset_task}-pretrain-checkpoint.pth')
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if args.projection:
                to_save.update({'W': W.state_dict()})
            torch.save(to_save, checkpoint_path)

        print('train_stats',train_stats)

    print(f'Best loss: {best_loss:4f} at epoch {best_epoch}')

    if not args.no_log:
        wandb.finish()
