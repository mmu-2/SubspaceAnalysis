import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models, datasets
from process_cubs import CUB, CUB_Image
import argparse
from PIL import Image
import os
import torch.nn as nn
import wandb
import numpy as np

from cubs_classification_dataset import CUBClassificationDataset

# python classification_pipeline.py --epochs 100 --k 32 --dataset fmnist --model rn18 --projection --experiment test --trial 1
# python classification_pipeline.py --epochs 10 --k 32 --dataset fmnist --model rn18 --projection --experiment test --trial 1 --no_log

def parse_args():
    parser = argparse.ArgumentParser(description="Process CUB_200_2011 dataset.")
    parser.add_argument('--data_dir', type=str, default="./CUB_200_2011/",
                        help='Directory where the CUB_200_2011 dataset is located.')
    parser.add_argument('--certainty_threshold', type=int, default=4,
                        help='Threshold for certainty levels in CUBS dataset.')
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--k', type=int, help="Dimension of latent space from projection")
    parser.add_argument('--dataset', type=str, choices=['fmnist', 'cub', 'cifar10'], required=True, help="Dataset being evaluated on.")
    parser.add_argument('--model', type=str, choices=['rn18', 'rn50'], default='rn18', help="Backbone model being evaluated on.")
    parser.add_argument('--projection', action='store_true', help="Flag that turns on projection bottleneck W")
    parser.add_argument('--experiment', type=str, default='default', help="Experiment name, use for grouping in wandb")
    parser.add_argument('--trial', type=int, default=1, help="Trial number. Useful for slurm purposes.")
    parser.add_argument('--no_log', action='store_true', help="Flag that turns off wandb. Useful during debugging.")
    return parser.parse_args()

def get_dataset(args):
    """
    Depending on the dataset selected, we return train_dataset, test_dataset with
    the correct transform preprocessing.
    """
    if args.dataset == 'cub':
        # CUBS has their preprocessing happening inside CUBClassificationDataset
        cub = CUB(args)
        train_dataset = CUBClassificationDataset(cub.CUB_train_set)
        test_dataset = CUBClassificationDataset(cub.CUB_val_set)

    if args.dataset == 'fmnist':
        transform = v2.Compose([
            v2.Grayscale(1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5])
        ])
        train_dataset = datasets.FashionMNIST(
            root='./fashionmnist',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./fashionmnist',
            train=False,
            download=True,
            transform=transform
        )

    # normalization constants coming from 
    # https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    if args.dataset == 'cifar10':
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.49139968, 0.48215841, 0.44653091],
                         [0.24703223, 0.24348513, 0.26158784]),
        ])
        train_dataset = datasets.CIFAR10(
            root='./cifar10',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./cifar10',
            train=False,
            download=True,
            transform=transform
        )
    
    # if args.dataset == 'celeba':
    #     transform = v2.Compose([
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         v2.Normalize([])
    #     ])
    
    return train_dataset, test_dataset

def get_model(args):
    """
    Depending on the dataset selected, we return the model modified for the corresponding dataset format.
    The model will not be moved to cuda.
    """

    if args.model == 'rn18':
        model = models.resnet18(weights=None)
        fc_in_layers = 512
    elif args.model == 'rn50':
        model = models.resnet50(weights=None)
        fc_in_layers = 2048
    else:
        raise ValueError()


    if args.dataset == 'cub':
        num_classes = 200
    elif args.dataset == 'fmnist':
        num_classes = 10
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #switch to single channel
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        raise ValueError()
    
    model.fc = nn.Linear(in_features=fc_in_layers, out_features=num_classes, bias=True)
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

    optimizer_parameters = [{'params': model.parameters(), 'lr': 0.001}]
    if args.projection:
        optimizer_parameters.append(
            {'params': W.parameters(), 'lr': 0.001}
        )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    best_acc = 0.0

    if not args.no_log:
        wandb.init(
            project="image_classification",
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
            }
        )

    
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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
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

                    if args.projection:
                        loss = loss + (W.weight @ W.weight.T - I).square().sum()
                        # loss = loss + (W.weight @ W.weight.T - I).abs().sum()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                optimizer.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # print(((W.weight @ W.weight.T) - I).square().sum())

            if phase == 'train':
                if not args.no_log:
                    wandb.log({
                        "epoch": epoch,
                        ("train/loss"): epoch_loss,
                        ("train/acc"): epoch_acc,
                    }, commit=False)
            else:
                if not args.no_log:
                    log = {
                        ("val/loss"): epoch_loss,
                        ("val/acc"): epoch_acc,
                    }
                    if args.projection:
                        log["W@W.T"] = (W.weight @ W.weight.T - I).square().sum()
                    wandb.log(log)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs(args.output, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_classification_model.pth'))
                if args.projection:
                    torch.save(W.state_dict(), os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_classification_projection.pth'))

    print(f'Best val Acc: {best_acc:4f}')

    # Log some sample images from the training dataset.
    if not args.no_log and args.projection:
        # load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_classification_model.pth')
        # model.load_state_dict(torch.load(load_path))
        # model.eval()
        load_path = os.path.join(args.output, f'{args.experiment}-{args.dataset}-{args.trial}_best_classification_projection.pth')
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
