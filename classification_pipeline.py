import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models, datasets
import argparse
from PIL import Image
import os
import torch.nn as nn
import wandb
import numpy as np
import timm
from mae.models_vit import vit_tiny_patch2, vit_base_patch2, vit_micro_patch2

from cubs_dataset import CUB, CUBClassificationDataset

# python classification_pipeline.py --epochs 100 --k 32 --dataset fmnist --model rn18 --projection --experiment test --trial 1
# python classification_pipeline.py --epochs 10 --k 32 --dataset fmnist --model rn18 --projection --experiment test --trial 1 --no_log

def train_one_epoch(args, model, W, criterion, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(True):
                    
            if args.projection:
                flattened = inputs.flatten(1)
                down_projected = flattened @ W.weight.T
                reconstructed = down_projected @ W.weight
                reshaped_input = reconstructed.reshape(inputs.shape)
                outputs = model(reshaped_input)
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if args.projection:
                loss = loss + args.projection_weight * (W.weight @ W.weight.T - I).square().sum()
                # loss = loss + (W.weight @ W.weight.T - I).abs().sum()

            loss = loss / args.accum_iter
            loss.backward()

            # weights update
            if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double().item() / len(dataloader.dataset)

    return {'loss': epoch_loss, 'acc': epoch_acc}

def evaluate(args, model, W, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
                    
            if args.projection:
                flattened = inputs.flatten(1)
                down_projected = flattened @ W.weight.T
                reconstructed = down_projected @ W.weight
                reshaped_input = reconstructed.reshape(inputs.shape)
                outputs = model(reshaped_input)
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double().item() / len(dataloader.dataset)

    return {'loss': epoch_loss, 'acc': epoch_acc}


def parse_args():
    parser = argparse.ArgumentParser(description="Classification dataset pipeline.")
    parser.add_argument('--data_dir', type=str, default=None, help='Directory where the dataset is located.')
    parser.add_argument('--certainty_threshold', type=int, default=4, help='Threshold for certainty levels in CUBS dataset.')
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--projection_lr', type=float, default=None, help="Learning rate for projection layer only.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--data_aug', action='store_true', help="Flag that turns on data augmentations.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--k', type=int, help="Dimension of latent space from projection")
    parser.add_argument('--dataset', type=str, choices=['fmnist', 'cub', 'cifar10', 'celeba', 'caltech101'], required=True, help="Dataset being evaluated on.")
    parser.add_argument('--model', type=str, 
                        choices=['rn18', 'rn34', 'rn50', 'rn101', 
                                 'vit2-mu', 'vit2-t','vit16-t', 'vit16-s', 'vit2-b', 'vit16-b', 'vit16-l', 'vit14-h'],
                        default='rn18', help="Backbone model being evaluated on.")
    parser.add_argument('--projection', action='store_true', help="Flag that turns on projection bottleneck W")
    parser.add_argument('--projection_weight', type=float, default=1.0, help="Loss weight of the regularization term.")
    parser.add_argument('--experiment', type=str, default='default', help="Experiment name, use for grouping in wandb")
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Useful for slurm purposes.")
    parser.add_argument('--no_log', action='store_true', help="Flag that turns off wandb. Useful during debugging.")
    return parser.parse_args()

def get_dataset(args, transform_train = None, transform_val = None):
    """
    Depending on the dataset selected, we return train_dataset, test_dataset with
    the correct transform preprocessing.

    It's expected that you pass in both transform_train and transform_val, or else you pass in neither.
    """

    # Set some default data root directories for datasets.
    if not args.data_dir:
        if args.dataset == 'cub': args.data_dir = './data/CUB_200_2011/'
        elif args.dataset == 'fmnist': args.data_dir = './data/fashionmnist/'
        elif args.dataset == 'cifar10': args.data_dir = './data/cifar10/'
        elif args.dataset == 'celeba': args.data_dir = './data/celeba/'
        elif args.dataset == 'caltech101': args.data_dir = './data/caltech101/'
        else:
            raise ValueError()

    if args.dataset == 'cub':
        # CUBS has their preprocessing happening inside CUBClassificationDataset
        cub = CUB(args)
        train_dataset = CUBClassificationDataset(cub.CUB_train_set)
        test_dataset = CUBClassificationDataset(cub.CUB_val_set)
        if transform_train and transform_val:
            train_dataset.transform = transform_train
            test_dataset.transform = transform_val
    elif args.dataset == 'fmnist':
        if not transform_train and not transform_val:
            transform_train = v2.Compose([
                v2.Grayscale(1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5])
            ])
            transform_val = transform_train
        train_dataset = datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        test_dataset = datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transform_val
        )
    # normalization constants coming from 
    # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    elif args.dataset == 'cifar10':
        if not transform_train and not transform_val:
            transform_val = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.49139968, 0.48215841, 0.44653091],
                            [0.24703223, 0.24348513, 0.26158784]),
            ])

            if args.data_aug:
                transform_train = v2.Compose([
                    v2.ToImage(),
                    v2.RandomHorizontalFlip(.5),
                    v2.ColorJitter(.2, .2, .2),
                    v2.RandomAutocontrast(.5),
                    v2.RandomCrop(32, 4),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.49139968, 0.48215841, 0.44653091],
                                [0.24703223, 0.24348513, 0.26158784]),
                ])
            else:
                transform_train = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.49139968, 0.48215841, 0.44653091],
                                [0.24703223, 0.24348513, 0.26158784]),
                ])

        train_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transform_val
        )

    elif args.dataset == 'celeba':
        if not transform_train and not transform_val:
            transform_train = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
            transform_val = transform_train

        train_dataset = datasets.CelebA(
            root=args.data_dir,
            split='train', #train, valid, test, all
            target_type='identity', #attr, identity, bbox, landmarks
            download=True,
            transform=transform_train
        )
        test_dataset = datasets.CelebA(
            root=args.data_dir,
            split='valid',
            target_type='identity',
            download=True,
            transform=transform_val
        )
        # This fixes the ids to be indexed by 0.
        # Hacky fix because it assumes knowledge of internal variables but it should work for our purposes.
        # train_dataset.identity = torch.tensor((train_dataset.identity - 1))
        train_dataset.identity = (train_dataset.identity - 1).clone().detach()
        test_dataset.identity = (test_dataset.identity - 1).clone().detach()
    elif args.dataset == 'caltech101':
        if not transform_train and not transform_val:
            transform_train = v2.Compose([
                v2.Resize(224),
                v2.CenterCrop(224), #variable sized images
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
            transform_val = transform_train
        train_dataset = datasets.Caltech101(
            root=args.data_dir,
            target_type='category',
            download=True,
            transform=transform_train
        )
        test_dataset = datasets.Caltech101(
            root=args.data_dir,
            target_type='category',
            download=True,
            transform=transform_val
        )
    else:
        raise ValueError()

    return train_dataset, test_dataset

def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """


    if args.dataset == 'cub':
        img_size = 224
        num_classes = 200
        in_chans = 3
    elif args.dataset == 'fmnist':
        img_size = 28
        num_classes = 10
        in_chans = 1
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #switch to single channel
    elif args.dataset == 'cifar10':
        img_size = 32
        num_classes = 10
        in_chans = 3
    elif args.dataset == 'celeba':
        img_size = 224
        num_classes = 10177
        in_chans = 3
    elif args.dataset == 'caltech101':
        img_size = 224
        num_classes = 101
        in_chans = 3
    else:
        raise ValueError()
    

    if args.model == 'rn18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.model == 'rn34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.model == 'rn50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.model =='rn101':
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)
        model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.model =='vit2-mu':
        model = vit_micro_patch2(num_classes = num_classes, global_pool = False, img_size=img_size, in_chans=in_chans)
    elif args.model == 'vit16-t':
        model = timm.create_model('vit_tiny_patch16_224', img_size=img_size, num_classes=num_classes, in_chans=in_chans)
    elif args.model == 'vit2-t':
        model = vit_tiny_patch2(num_classes = num_classes, global_pool = False, img_size=img_size, in_chans=in_chans)
    elif args.model == 'vit16-s':
        model = timm.create_model('vit_small_patch16_224', img_size=img_size, num_classes=num_classes, in_chans=in_chans)
    elif args.model == 'vit2-b':
        model = vit_base_patch2(num_classes = num_classes, global_pool = False, img_size=img_size, in_chans=in_chans)
    elif args.model == 'vit16-b':
        model = timm.create_model('vit_base_patch16_224', img_size=img_size, num_classes=num_classes, in_chans=in_chans)
    elif args.model == 'vit16-l':
        model = timm.create_model('vit_large_patch16_224', img_size=img_size, num_classes=num_classes, in_chans=in_chans)
    elif args.model == 'vit14-h':
        model = timm.create_model('vit_huge_patch14_224', img_size=img_size, num_classes=num_classes, in_chans=in_chans)
    else:
        raise ValueError()

    return model

if __name__ == '__main__':
    
    args = parse_args()

    train_dataset, test_dataset = get_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.projection:
        # set bias=False is equivalent to creating a Parameters layer
        # This way, I can initialize cleanly without using custom code.
        W = torch.nn.Linear(train_dataset[0][0].flatten().shape[0], args.k, bias=False, device=device)
        I = torch.eye(args.k, device=device)

    optimizer_parameters = [{'params': model.parameters(), 'lr': args.lr}]
    if args.projection:
        lr = args.projection_lr if args.projection_lr else args.lr
        optimizer_parameters.append(
            {'params': W.parameters(), 'lr': lr}
        )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    if not args.no_log:
        wandb.init(
            project="image_classification",
            name=f'{args.experiment}-{args.dataset}-{args.trial}',
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
        )
        print(args)

    best_acc = 0.0
    best_epoch = 0
    dataloaders = {
        'train': train_loader,
        'val': test_loader
    }
    dataset_sizes ={
        'train': len(train_dataset),
        'val': len(test_dataset)
    }
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'epoch {epoch}')

        if args.projection:
            train_stats = train_one_epoch(args, model, W, criterion, train_loader, optimizer, device)
            val_stats = evaluate(args, model, W, criterion, test_loader, device)
        else:
            train_stats = train_one_epoch(args, model, None, criterion, train_loader, optimizer, device)
            val_stats = evaluate(args, model, None, criterion, test_loader, device)

        print(f'train Loss: {train_stats['loss']:.4f} Acc: {train_stats['acc']:.4f}')
        print(f'val Loss: {val_stats['loss']:.4f} Acc: {val_stats['acc']:.4f}')

        if not args.no_log:
            log = {'epoch': epoch,}
            for k,v in train_stats.items():
                log.update({f'train/{k}': v})
            for k,v in val_stats.items():
                log.update({f'val/{k}': v})

            if args.projection:
                log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum().item()
            print('log',log)
            wandb.log(log)

        # save best model
        if val_stats['acc'] > best_acc:
            best_acc = val_stats['acc']
            best_epoch = epoch
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.k}_best_classification_model.pth'))
            if args.projection:
                torch.save(W.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.k}_best_classification_projection.pth'))

    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')

    # Log some sample images from the training dataset.
    if not args.no_log and args.projection:
        # load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_classification_model.pth')
        # model.load_state_dict(torch.load(load_path))
        # model.eval()
        load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.k}_best_classification_projection.pth')
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
            np_reconstructed_image = reshaped_input.detach().cpu().numpy()
            np_image = image.detach().cpu().numpy()
            wandb.log({
                "Original": wandb.Image(np.transpose(np_image,(1,2,0))),
                "Image @ W.T @ W": wandb.Image(np.transpose(np_reconstructed_image, (1,2,0)))
            })

    if not args.no_log:
        wandb.finish()
