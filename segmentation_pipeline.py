import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import argparse
import wandb
import os

from cubs_dataset import CUB, CUBSegmentationDataset

def get_dataset(args, transform_train = None, transform_val = None):
    """
    Depending on the dataset selected, we return train_dataset, test_dataset with
    the correct transform preprocessing.
    """
    if not args.data_dir:
        if args.dataset == 'cub': args.data_dir = './CUB_200_2011/'
        else:
            raise ValueError()

    if args.dataset == 'cub':
        cub = CUB(args)
        train_dataset = CUBSegmentationDataset(cub.CUB_train_set)
        test_dataset = CUBSegmentationDataset(cub.CUB_val_set)
        classes = {0: 'Background', 1: 'Bird'}
        if transform_train and transform_val:
            train_dataset.transform = transform_train
            test_dataset.transform = transform_val
    else:
        raise ValueError()

    return train_dataset, test_dataset, classes

def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """

    if args.dataset == 'cub':
        num_classes = 2
    else:
        raise ValueError()

    if args.model == 'dlv3rn50':
        model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    else:
        raise ValueError()

    return model

def train_one_epoch(model, W, optimizer, criterion, dataloader, device):
    model.train()
    
    running_loss = 0.0
    for batch_idx, (images, target_masks) in enumerate(dataloader):
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

        loss = criterion(outputs, target_masks.to(torch.float))
        if args.projection:
            loss = loss + (W.weight @ W.weight.T - I).square().sum()
            # loss = loss + (W.weight @ W.weight.T - I).abs().sum()
        loss = loss / args.accum_iter
        running_loss += loss.item()
        loss.backward()
        # weights update
        if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()
    return {'loss': running_loss}

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

    return {'acc': pixel_accuracy, 'mIoU': mIoU}



def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation dataset pipeline")
    parser.add_argument('--data_dir', type=str, default=None, help='Directory where the dataset is located.')
    parser.add_argument('--certainty_threshold', type=int, default=4, help='Threshold for certainty levels in CUBS dataset.')
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="SGD with weight decay")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--k', type=int, help="Dimension of latent space from projection")
    parser.add_argument('--dataset', type=str, choices=['cub', 'celebamask'], required=True, help="Dataset being evaluated on.")
    parser.add_argument('--model', type=str, choices=['dlv3rn50'], default='dlv3rn50', help="Backbone model being evaluated on.")
    parser.add_argument('--projection', action='store_true', help="Flag that turns on projection bottleneck W")
    parser.add_argument('--experiment', type=str, default='default', help="Experiment name, use for grouping in wandb")
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Useful for slurm purposes.")
    parser.add_argument('--no_log', action='store_true', help="Flag that turns off wandb. Useful during debugging.")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    train_dataset, test_dataset, classes = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    criterion = nn.CrossEntropyLoss(reduction='mean') #in this case, I think it's okay to include the background
    if args.projection:
        W = torch.nn.Linear(train_dataset[0][0].flatten().shape[0], args.k, bias=False, device=device)
        I = torch.eye(args.k, device=device)

    optimizer_parameters = [{'params': model.parameters(), 'lr': args.lr, 'momentum':args.momentum, 'weight_decay':args.weight_decay}]
    if args.projection:
        optimizer_parameters.append(
            {'params': W.parameters(), 'lr': args.lr}
        )
    optimizer = torch.optim.SGD(optimizer_parameters)
    
    if not args.no_log:
        wandb.init(
            project="image_segmentation",
            name=f'{args.experiment}-{args.dataset}-{args.trial}',
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "accum_iter": args.accum_iter,
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
            train_stats = train_one_epoch(model, W, optimizer, criterion, train_loader, device)
            val_stats = evaluate(model, W, test_loader, device=device)
        else:
            train_stats = train_one_epoch(model, None, optimizer, criterion, train_loader, device)
            val_stats = evaluate(model, None, test_loader, device=device)

        if not args.no_log:
            log = {'epoch': epoch,}
            for k,v in train_stats.items():
                log.update({f'train/{k}': v})
            for k,v in val_stats.items():
                log.update({f'val/{k}': v})

            if args.projection:
                log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum().item()
            wandb.log(log)

        # deep copy the model
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            best_epoch = epoch
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_segmentation_model.pth'))
            if args.projection:
                torch.save(W.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_segmentation_projection.pth'))

    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

    # Log some sample images from the training dataset.
    if not args.no_log and args.projection:
        load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_segmentation_projection.pth')
        W.load_state_dict(torch.load(load_path))
        W.eval()
        for i in range(2):
            image, _ = train_dataset[i] #indexing from dataset will not have batch dimension
            image = image.to(device)
            W.eval()

            flattened = image.flatten()
            down_projected = flattened @ W.weight.T
            reconstructed = down_projected @ W.weight
            reshaped_input = reconstructed.reshape(image.shape)
            outputs = model(reshaped_input.unsqueeze(0))['out']
            pred = torch.argmax(outputs, dim=1).squeeze().detach().cpu().numpy()
            
            np_reconstructed_image = reshaped_input.detach().cpu().numpy()
            np_image = image.detach().cpu().numpy()
            

            wandb.log({
                "Original": wandb.Image(np.transpose(np_image,(1,2,0))),
                "Image @ W.T @ W": wandb.Image(np.transpose(np_reconstructed_image, (1,2,0))),
                "Mask on Original Image": wandb.Image(
                    np.transpose(np_image, (1,2,0)), masks={
                        "predictions" : {
                        "mask_data" : pred,
                        "class_labels" : classes
                        }
                    }),
                "Mask on Reconstructed Image": wandb.Image(
                    np.transpose(np_reconstructed_image, (1,2,0)), masks={
                        "predictions" : {
                        "mask_data" : pred,
                        "class_labels" : classes
                        }
                    }),
            })

    if not args.no_log:
        wandb.finish()
