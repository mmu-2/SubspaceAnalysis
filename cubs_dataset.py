import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from process_cubs import CUB, CUB_Image
from PIL import Image

class CUBClassificationDataset(Dataset):

    def __init__(self, cub_images:list[CUB_Image]):
        self.image_paths = []
        self.label_ids = []
        for cub_image in cub_images:
            self.image_paths.append(cub_image.img_path)
            self.label_ids.append(cub_image.label - 1) #Note I'm subtracting 1 here because PyTorch wants index from 0.
        self.transform = v2.Compose([
            # Random resize crop is standard, but I'm concerned that the birds task is too fine-grained for that.
            v2.Resize(224),
            v2.CenterCrop(224),
            # v2.RandomResizedCrop(224),
            # v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.4825, 0.4904, 0.4227], std=[0.2295, 0.2250, 0.2597]) #Cubs normalization values I calculated
            v2.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #Imagenet values
        ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.image_paths[idx])), torch.tensor(self.label_ids[idx])


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

        class1 = mask >= self.segmentation_threshold
        background_mask = mask < self.segmentation_threshold
        label = torch.tensor(self.label_ids[idx])
        target = {}
        target['masks'] = tv_tensors.Mask(torch.cat([background_mask, class1], dim=0))
        # print('target[masks].shape',target['masks'].shape)
        target['labels'] = label
        target['image_id'] = self.image_ids[idx] # this might need to convert to tensor?
        target['area'] = mask.sum()
        target['iscrowd'] = False
        img, target = self.transform(img, target)

        if target['masks'].shape != (2, 224, 224):
            print('error at',self.image_ids[idx])
            print('target[masks].shape',target['masks'].shape)
            print('mask.shape',mask.shape)
            print('self.image_paths[idx]',self.image_paths[idx])
            print('self.mask_paths[idx]',self.mask_paths[idx])
        return img, target