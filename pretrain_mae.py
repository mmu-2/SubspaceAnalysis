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

from mae.models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from mae.utils import add_weight_decay
from mae.utils import NativeScalerWithGradNormCount as NativeScaler
from mae.utils import save_model, load_model, adjust_learning_rate, SmoothedValue

from PIL import Image



def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """

    if args.model == 'mae_vit_base_patch16':
        model = mae_vit_base_patch16(norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_vit_large_patch16':
        model = mae_vit_large_patch16(norm_pix_loss=args.norm_pix_loss)
    elif args.model == 'mae_vit_huge_patch14':
        model = mae_vit_huge_patch14(norm_pix_loss=args.norm_pix_loss)
    else:
        raise ValueError()
    
    return model

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

        if W:
            flattened = images.flatten(1)
            down_projected = flattened @ W.weight.T
            reconstructed = down_projected @ W.weight
            images = reconstructed.reshape(images.shape)

        with torch.cuda.amp.autocast():
            loss, y, mask = model(images, mask_ratio=args.mask_ratio)


        # y = model.unpatchify(y)
        # y = torch.einsum('nchw->nhwc', y).detach().cpu()
        # # visualize the mask
        # mask = mask.detach()
        # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        # mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        # x = torch.einsum('nchw->nhwc', x)
        # # masked image
        # im_masked = x * (1 - mask)
        # # MAE reconstruction pasted with visible patches
        # im_paste = x * (1 - mask) + y * mask
        # # make the plt figure larger
        # plt.rcParams['figure.figsize'] = [24, 24]
        # plt.subplot(1, 4, 1)
        # def show_image(image, title=''):
        #     imagenet_mean = np.array([0.485, 0.456, 0.406])
        #     imagenet_std = np.array([0.229, 0.224, 0.225])
        #     # image is [H, W, 3]
        #     assert image.shape[2] == 3
        #     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        #     plt.title(title, fontsize=16)
        #     plt.axis('off')
        #     return
        # show_image(x[0], "original")

        
        x = images.detach().cpu()
        # assuming x, y, and mask are already defined
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        x = torch.einsum('nchw->nhwc', x)
        # masked image
        im_masked = x * (1 - mask)
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        # function to save image without using plt
        def save_image(image_tensor, filename):
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            # image is [H, W, 3], we normalize and convert to uint8
            image = torch.clip((image_tensor * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
            # convert to PIL image and save
            image_pil = Image.fromarray(image.astype(np.uint8))
            image_pil.save(filename)

        # save the original image
        # if data_iter_step == 0:
        #     save_image(x[0], f'original_image{epoch}.png')
        #     save_image(im_masked[0], f'masked{epoch}.png')
        #     save_image(y[0], f'reconstruction{epoch}.png')
        #     save_image(im_paste[0], f'reconstruction_plus_visible{epoch}.png')

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        running_loss += loss
    lr = optimizer.param_groups[0]["lr"]

    return {'running_loss': running_loss, 'lr': lr}

@torch.inference_mode()
def evaluate(model, W, dataloader, device):
    #Find evaluation metrics:
    #ADE20K:
    # - Pixel accuracy: proportion of correctly classified pixels
    # - Mean accuracy: the proportion of correctly classified pixels averaged over all the classes
    # - Mean IoU: IoU between pred and GT pixels, averaged over all classes
    # - Weighted IoU: IoU weighted by the total pixel ratio of the class (i'm guessing if more of a class in an image, weight it more.)
    # The formula for pixel accuracy seems to change depending on who implemented it...
    #PASCAL VOC:
    # mIoU: averaged across 21 classes

    intersections = 0
    unions = 0

    model.eval()

    for images, target_masks in dataloader:
        images = images.to(device)
        target_masks = target_masks.to(device)
        
        if W:
            flattened = images.flatten(1)
            down_projected = flattened @ W.weight.T
            reconstructed = down_projected @ W.weight
            reshaped_input = reconstructed.reshape(images.shape)
            outputs = model(reshaped_input)['out']
        else:
            outputs = model(images)['out'] # [bs, classes, 224, 224]

        pred = torch.argmax(outputs, dim=1)  # shape: [bs, 224, 224]

        # Create binary masks for each class
        num_classes = outputs.shape[1]
        masks = (pred[:, None, ...] == torch.arange(num_classes, device=device)[None, :, None, None]).float()  # shape: [bs, classes, 224, 224]
        intersection = (masks * target_masks).sum(dim=(0,2,3))
        intersections += intersection
        unions += (masks.sum(dim=(0,2,3)) + target_masks.sum(dim=(0,2,3))) - intersection

    image_batch, _ = next(iter(dataloader))
    image_batch = torch.stack([image.to(device) for image in image_batch])

    pixel_accuracy = intersections.sum() / (len(dataloader.dataset) * 1 * image_batch.shape[2] * image_batch.shape[3])
    print(f'pixel accuracy: {pixel_accuracy * 100:.2f}')

    mIoU = (intersections / unions).sum() / len(intersections)
    print(f'mIoU: {mIoU}')

    # return pixel_accuracy, mIoU
    return {'acc': pixel_accuracy, 'mIoU': mIoU}




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


    
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--model', type=str, 
                        choices=['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'],
                        default='mae_vit_base_patch16', 
                        help="Backbone model being evaluated on.")
    parser.add_argument('--resume', default='', help='resume from checkpoint')

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
            v2.RandomResizedCrop((224, 224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if args.dataset_task == 'classification':
        # TODO: transform_val doesn't matter at the moment because we don't use it.
        train_dataset, test_dataset = get_classification_dataset(args, transform_train==transform_train, transform_val=transform_train)
    else:
        train_dataset, test_dataset, classes = get_segmentation_dataset(args, transform_train=transform_train, transform_val=transform_train)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    # criterion = nn.CrossEntropyLoss(reduction='mean') #in this case, I think it's okay to include the background
    if args.projection:
        W = torch.nn.Linear(train_dataset[0][0].flatten().shape[0], args.k, bias=False, device=device)
        W.load_state_dict(torch.load(args.w_weights))
        I = torch.eye(args.k, device=device)

    # optimizer_parameters = [{'params': model.parameters(), 'lr': args.lr, 'momentum':args.momentum, 'weight_decay':args.weight_decay}]
    # if args.projection:
    #     optimizer_parameters.append(
    #         {'params': W.parameters(), 'lr': args.lr}
    #     )
    # optimizer = torch.optim.SGD(optimizer_parameters)
    param_groups = add_weight_decay(model, args.weight_decay)

    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # Don't plan to train W at this time.
    # if args.projection:
    #     param_groups.append({'params': W.parameters(), 'weight_decay': args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()
    load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # List to store saved checkpoints
    saved_checkpoints = []
    
    if not args.no_log:
        wandb.init(
            project="mae_pretrain",
            name=f'{args.experiment}-{args.dataset}-{args.trial}',
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

    best_loss = 99999
    best_epoch = 0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'epoch: {epoch}')
        if args.projection:
            train_stats = train_one_epoch(model, W, train_loader, optimizer, device, epoch, loss_scaler, args=args)
            # val_stats = evaluate(model, W, test_loader, device=device)
        else:
            train_stats = train_one_epoch(model, None, train_loader, optimizer, device, epoch, loss_scaler, args=args)
            # val_stats = evaluate(model, None, test_loader, device=device)
    
        if not args.no_log:
            log = {'epoch': epoch,}
            for k,v in train_stats.items():
                log.update({f'train/{k}': v})
            # for k,v in val_stats.items():
            #     log.update({f'val/{k}': v})

            if args.projection:
                log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum()
            wandb.log(log)
        

        if train_stats['running_loss'] < best_loss:
            best_loss = train_stats['running_loss']
            best_epoch = epoch
            os.makedirs(args.output, exist_ok=True)
            save_model(args, epoch, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

            output_dir = Path(args.output)
            epoch_name = str(epoch)
            checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
            # Append the saved checkpoint to the list
            saved_checkpoints.append(checkpoint_path)
            if len(saved_checkpoints) > 3:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
        print(train_stats)
        # if val_stats['acc'] > best_acc:
        #     best_acc = val_stats['acc']
        #     best_epoch = epoch
        #     os.makedirs(args.output, exist_ok=True)
        #     save_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f'Best loss: {best_loss:4f} at epoch {best_epoch}')

    # Log some sample images from the training dataset.
    # if not args.no_log and args.projection:
    #     load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_segmentation_projection.pth')
    #     W.load_state_dict(torch.load(load_path))
    #     W.eval()
    #     for i in range(2):
    #         image, _ = train_dataset[i] #indexing from dataset will not have batch dimension
    #         image = image.to(device)
    #         W.eval()

    #         flattened = image.flatten()
    #         down_projected = flattened @ W.weight.T
    #         reconstructed = down_projected @ W.weight
    #         reshaped_input = reconstructed.reshape(image.shape)
    #         outputs = model(reshaped_input.unsqueeze(0))['out']
    #         pred = torch.argmax(outputs, dim=1).squeeze().detach().cpu().numpy()
            
    #         np_reconstructed_image = reshaped_input.detach().cpu().numpy()
    #         np_image = image.detach().cpu().numpy()
            

    #         wandb.log({
    #             "Original": wandb.Image(np.transpose(np_image,(1,2,0))),
    #             "Image @ W.T @ W": wandb.Image(np.transpose(np_reconstructed_image, (1,2,0))),
    #             "Mask on Original Image": wandb.Image(
    #                 np.transpose(np_image, (1,2,0)), masks={
    #                     "predictions" : {
    #                     "mask_data" : pred,
    #                     "class_labels" : classes
    #                     }
    #                 }),
    #             "Mask on Reconstructed Image": wandb.Image(
    #                 np.transpose(np_reconstructed_image, (1,2,0)), masks={
    #                     "predictions" : {
    #                     "mask_data" : pred,
    #                     "class_labels" : classes
    #                     }
    #                 }),
    #         })

    if not args.no_log:
        wandb.finish()
