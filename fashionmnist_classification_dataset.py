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


def parse_args():
    parser = argparse.ArgumentParser(description="Process CUB_200_2011 dataset.")
    parser.add_argument('--data_dir', type=str, default="./CUB_200_2011/",
                        help='Directory where the CUB_200_2011 dataset is located.')
    parser.add_argument('--certainty_threshold', type=int, default=4,
                        help='Threshold for certainty levels.')
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    parser.add_argument('--k', type=int, default=3000, help="Dimension of latent space from projection")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    # cub = CUB(args)
    # train_dataset = CUBClassificationDataset(cub.CUB_train_set)
    # test_dataset = CUBClassificationDataset(cub.CUB_val_set)
    # train_dataset = datasets.Caltech256('./Caltech256',transform=v2.Compose([
    #         #For now, do nothing else because other augmentations
    #         v2.RandomResizedCrop(224),
    #         v2.RandomHorizontalFlip(),
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         # v2.Normalize(mean=[0.4825, 0.4904, 0.4227], std=[0.2295, 0.2250, 0.2597]) #Cubs normalization values I calculated
    #         v2.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #Imagenet values
    #     ]),
    #     download=True)

    transform = v2.Compose([
            #For now, do nothing else because other augmentations
            # v2.RandomResizedCrop(224),
            # v2.RandomHorizontalFlip(),
            v2.Grayscale(1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.4825, 0.4904, 0.4227], std=[0.2295, 0.2250, 0.2597]) #Cubs normalization values I calculated
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #Imagenet values
            v2.Normalize([0.5], [0.5])
        ])
    train_dataset = datasets.FashionMNIST(
        root="./fashionmnist",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./fashionmnist",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    # num_classes = len(cub.classes)
    num_classes = 10
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #switch to single channel

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    W = torch.nn.Linear(28*28*1, args.k, bias=False, device=device) #set bias=False is equivalent to Parameters, but I don't have to initialize
    I = torch.eye(args.k, device=device)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 0.001},
        {'params': W.parameters(), 'lr': 0.001},
    ])
    
    best_acc = 0.0

    wandb.init(
        project="image_classification",
        name="single-channel-no-projection-1",
        config={
            "learning_rate": args.lr,
            "architecture": "resnet18",
            "dataset": "FashionMNIST",
            "epochs": args.epochs,
            "K": args.k,
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
                
                # import numpy as np
                # print('inputs.shape',inputs.shape)
                # # np_inputs = np.array(inputs[0])
                # np_inputs = np.uint8(inputs[0]*255)
                # np_inputs = np.transpose(np_inputs, (1,2,0))
                # print('np_inputs.shape',np_inputs.shape)
                # Image.fromarray(np_inputs).save('sample[0].png')
                # print('labels',labels)
                # exit()

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    flattened = inputs.flatten(1)
                    down_projected = flattened @ W.weight.T
                    reconstructed = down_projected @ W.weight
                    reshaped_input = reconstructed.reshape(inputs.shape)
                    outputs = model(reshaped_input)

                    # outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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
                wandb.log({
                    "epoch": epoch,
                    ("train/loss"): epoch_loss,
                    ("train/acc"): epoch_acc,
                    ("train/W@W.T"): (W.weight @ W.weight.T - I).square().sum(),
                }, commit=False)
            else:
                wandb.log({
                    ("val/loss"): epoch_loss,
                    ("val/acc"): epoch_acc,
                    ("val/W@W.T"): (W.weight @ W.weight.T - I).square().sum(),
                })

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs(args.output, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output, 'fmnist_classification_1channel_best_model2.pth'))
                torch.save(W.state_dict(), os.path.join(args.output, 'fmnist_classification_1channel_best_projection2.pth'))

    print(f'Best val Acc: {best_acc:4f}')
    wandb.finish()
