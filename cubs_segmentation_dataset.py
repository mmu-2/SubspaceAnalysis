import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import models, tv_tensors
from process_cubs import CUB, CUB_Image
import argparse
import torch.nn as nn
import wandb
from torchvision.io import read_image, ImageReadMode
import os

class CUBSegmentationDataset(Dataset):

    def __init__(self, cub_images:list[CUB_Image]):
        self.image_paths = []
        self.mask_paths = []
        self.label_ids = []
        self.image_ids = []
        self.segmentation_threshold = cub_images[0].segmentation_threshold
        for cub_image in cub_images:
            self.image_paths.append(cub_image.img_path)
            self.mask_paths.append(cub_image.seg_path)
            #TODO: The classification model is only doing okay at the moment. I suspect instance segmentation will be too hard
            # consider switching to just segmenting birds and this will still be a viable task.
            self.label_ids.append(cub_image.label) #Note I'm not subtracting 1 here because PyTorch wants index 0 to be background
            self.image_ids.append(cub_image.id)
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
        img = read_image(self.image_paths[idx])
        img = tv_tensors.Image(img)
        mask = read_image(self.mask_paths[idx], ImageReadMode.GRAY) # the occasional image is gray-alpha

        mask = mask >= self.segmentation_threshold
        label = torch.tensor(self.label_ids[idx])
        target = {}
        target['masks'] = tv_tensors.Mask(mask)
        target['labels'] = label
        target['image_id'] = self.image_ids[idx] # this might need to convert to tensor?
        target['area'] = mask.sum()
        target['iscrowd'] = False
        img, target = self.transform(img, target)

        if target['masks'].shape != (1, 224, 224):
            print('error at',self.image_ids[idx])
            print('target[masks].shape',target['masks'].shape)
            print('mask.shape',mask.shape)
            print('self.image_paths[idx]',self.image_paths[idx])
            print('self.mask_paths[idx]',self.mask_paths[idx])
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

import torch

def train_one_epoch(model, optimizer, criterion, data_loader, device):
    model.train()
    
    running_loss = 0.0
    for images, targets in data_loader:
        images = torch.stack([image.to(device) for image in images])
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        out = model(images)['out'] # [bs, classes, 224, 224]

        batched_masks = torch.stack([target['masks'] for target in targets])
        background_masks = ~batched_masks
        batched_masks = torch.cat((background_masks, batched_masks), dim=1)

        loss = criterion(out, batched_masks.to(torch.float))
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss

@torch.inference_mode()
def evaluate(model, data_loader, device):
    #Find evaluation metrics:
    #ADE20K:
    # - Pixel accuracy: proportion of correctly classified pixels
    # - Mean accuracy: the proportion of correctly classified pixels averaged over all the classes
    # - Mean IoU: IoU between pred and GT pixels, averaged over all classes
    # - Weighted IoU: IoU weighted by the total pixel ratio of the class (i'm guessing if more of a class in an image, weight it more.)
    # The formula for pixel accuracy seems to change depending on who implemented it...
    #PASCAL VOC:
    # mIoU: averaged across 21 classes

    class1_intersection = 0
    class1_union = 0
    background_intersection = 0
    background_union = 0
    model.eval()

    for images, targets in data_loader:
        images = torch.stack([image.to(device) for image in images])
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        out = model(images)['out'] # [bs, classes, 224, 224]
        pred = torch.argmax(out, dim=1)  # shape: [bs, 224, 224]

        # Create binary masks for each class
        mask0 = (pred == 0).unsqueeze(1).float()  # Binary mask for class 0, shape: [bs, 1, 224, 224]
        mask1 = (pred == 1).unsqueeze(1).float()  # Binary mask for class 1, shape: [bs, 1, 224, 224]

        batched_masks = torch.stack([target['masks'] for target in targets])
        background_masks = ~batched_masks

        intersection = (mask1 * batched_masks).sum()
        class1_intersection += intersection
        class1_union += (mask1.sum() + batched_masks.sum()) - intersection # - class1_intersection

        intersection = (mask0 * background_masks).sum()
        background_intersection += intersection
        background_union += (mask0.sum() + background_masks.sum()) - intersection #background_intersection

    image_batch, target_batch = next(iter(data_loader)) #len 32
    image_batch = torch.stack([image.to(device) for image in image_batch])

    pixel_accuracy = (class1_intersection + background_intersection) / (len(data_loader) * 1 * image_batch.shape[2] * image_batch.shape[3])
    print(f'pixel accuracy: {pixel_accuracy * 100:.2f}')

    mIoU = 0.5 * class1_intersection / class1_union + 0.5 * background_intersection / background_union
    print(f'mIoU: {mIoU}')

    # from PIL import Image
    # # Convert tensor to a NumPy array
    # mask_np = (mask0[-1].float()*255).squeeze(1).to(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()  # Use .cpu() if the tensor is on GPU
    # img = Image.fromarray(mask_np)
    # img.save('pred_mask0.png')
    # mask_np = (mask1[-1].float()*255).squeeze(1).to(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()  # Use .cpu() if the tensor is on GPU
    # img = Image.fromarray(mask_np)
    # img.save('pred_mask1.png')
    # mask_np = (background_masks[-1].float()*255).squeeze(1).to(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
    # img = Image.fromarray(mask_np)
    # img.save('GT_mask0.png')
    # mask_np = (batched_masks[-1].float()*255).squeeze(1).to(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
    # img = Image.fromarray(mask_np)
    # img.save('GT_mask1.png')

    return pixel_accuracy, mIoU



def parse_args():
    parser = argparse.ArgumentParser(description="Process CUB_200_2011 dataset.")
    parser.add_argument('--data_dir', type=str, default="./CUB_200_2011/",
                        help='Directory where the CUB_200_2011 dataset is located.')
    parser.add_argument('--certainty_threshold', type=int, default=4,
                        help='Threshold for certainty levels.')
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="SGD with weight decay")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--output', type=str, default='./model_weights', help="Output path of model checkpoints")
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    cub = CUB(args)

    train_dataset = CUBSegmentationDataset(cub.CUB_train_set)
    test_dataset = CUBSegmentationDataset(cub.CUB_val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # num_classes = len(cub.classes)
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

    model = model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    criterion = nn.CrossEntropyLoss(reduction='mean') #in this case, I think it's okay to include (photos are like 50% bird)

    wandb.init(
        project="image_segmentation",
        config={
            "learning_rate": args.lr,
            "architecture": "deeplabv3_resnet50",
            "dataset": "CUB_200_2011",
            "epochs": args.epochs,
        }
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'epoch: {epoch}')
        epoch_running_train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        accuracy, mIoU = evaluate(model, test_loader, device=device)
    
        wandb.log({
            "epoch": epoch,
            ("train/loss"): epoch_running_train_loss,
            ("val/acc"): accuracy,
            ("val/mIoU"): mIoU,
        })

        # deep copy the model
        if accuracy > best_acc:
            best_acc = accuracy
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, 'segmentation_best_model.pth'))
    wandb.finish()
