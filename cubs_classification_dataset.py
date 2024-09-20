import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models
from process_cubs import CUB, CUB_Image
import argparse
from PIL import Image
import os
import torch.nn as nn
import wandb

class CUBClassificationDataset(Dataset):

    def __init__(self, cub_images:list[CUB_Image]):
        self.image_paths = []
        self.label_ids = []
        for cub_image in cub_images:
            self.image_paths.append(cub_image.img_path)
            self.label_ids.append(cub_image.label - 1) #Note I'm subtracting 1 here because PyTorch wants index from 0.
        self.transform = v2.Compose([
            #For now, do nothing else because other augmentations
            v2.RandomResizedCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4825, 0.4904, 0.4227], std=[0.2295, 0.2250, 0.2597])
        ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx])), torch.tensor(self.label_ids[idx])

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
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    cub = CUB(args)

    train_dataset = CUBClassificationDataset(cub.CUB_train_set)
    test_dataset = CUBClassificationDataset(cub.CUB_val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    num_classes = len(cub.classes)
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0

    wandb.init(
        project="image_classification",
        config={
            "learning_rate": args.lr,
            "architecture": "resnet18",
            "dataset": "CUB_200_2011",
            "epochs": args.epochs,
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
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

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

            if phase == 'train':
                wandb.log({
                    "epoch": epoch,
                    ("train/loss"): epoch_loss,
                    ("train/acc"): epoch_acc,
                }, commit=False)
            else:
                wandb.log({
                    ("val/loss"): epoch_loss,
                    ("val/acc"): epoch_acc,
                })

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs(args.output, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output, 'best.pth'))

    print(f'Best val Acc: {best_acc:4f}')
    wandb.finish()
