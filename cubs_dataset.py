import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from PIL import Image

import cv2
import os
import copy
from typing import List
import numpy as np
import pandas as pd

import torch
from torchvision.transforms import v2


certainties_mapping = {
    1: 'not visible',
    2: 'guessing',
    3: 'probably',
    4: 'definitely'
}

parts_mapping = {
    1: 'back',
    2: 'beak',
    3: 'belly',
    4: 'breast',
    5: 'crown',
    6: 'forehead',
    7: 'left eye',
    8: 'left leg',
    9: 'left wing',
    10: 'nape',
    11: 'right eye',
    12: 'right leg',
    13: 'right wing',
    14: 'tail',
    15: 'throat'
}

def load_data(args):
    df = pd.read_csv(os.path.join(args.data_dir, 'classes.txt'), 
                    delimiter=' ', 
                    names=['class_id', 'class_name'])
    class_2_name = dict(zip(df['class_id'], df['class_name']))

    df = pd.read_csv(os.path.join(args.data_dir, 'images.txt'), 
                    delimiter=' ', 
                    names=['image_id', 'path'])
    image_id_2_image_path = dict(zip(df['image_id'], df['path']))
    image_id_2_seg_path = copy.deepcopy(image_id_2_image_path)
    for key in image_id_2_seg_path:
        image_id_2_seg_path[key] = image_id_2_seg_path[key].replace('.jpg', '.png')

    df = pd.read_csv(os.path.join(args.data_dir, 'image_class_labels.txt'), 
                    delimiter=' ', 
                    names=['image_id', 'class'])
    image_id_2_class = dict(zip(df['image_id'], df['class']))

    df = pd.read_csv(os.path.join(args.data_dir, 'bounding_boxes.txt'), 
                    delimiter=' ',
                    names=['image_id', 'x', 'y', 'width', 'height'])
    keys = list(df['image_id'])
    values = list(df[['x', 'y', 'width', 'height']].apply(tuple, axis=1))
    image_id_2_bbox = dict(zip(keys, values))

    df = pd.read_csv(os.path.join(args.data_dir, 'train_test_split.txt'), 
                    delimiter=' ', 
                    names=['image_id', 'split'])
    image_id_2_split = dict(zip(df['image_id'], df['split']))

    #####attributes#####
    df = pd.read_csv(os.path.join(args.data_dir, 'attributes.txt'), 
                    delimiter=' ', 
                    names=['attr_id', 'attr_name'])
    attr_id_2_name = dict(zip(df['attr_id'], df['attr_name']))
    # print(attr_id_2_name)

    df = pd.read_csv(os.path.join(args.data_dir, 'attributes', 'class_attribute_labels_continuous.txt'),
                    delimiter=' ',
                    names=list(sorted(attr_id_2_name.keys())))
    class_id_2_attr_makeup = dict(zip(df.index + 1, df.values))


    image_id_and_attr_id_2_certainty_id = dict()
    with open(os.path.join(args.data_dir, 'attributes', 'image_attribute_labels.txt'), 'r') as file:
        for line_num, line in enumerate(file):
            # process each line here
            ary = line.split()
            if len(ary) == 5:
                image_id, attr_id, is_present, certainty_id, time = ary
            elif len(ary) == 6: # it looks like there is sometimes an extra " 0 " inserted at position 4
                image_id, attr_id, is_present, certainty_id, _test, time = ary
                assert _test == '0'
            else:
                raise ValueError(f"Unexpected number of elements in line: {line_num}")
            image_id_and_attr_id_2_certainty_id[(int(image_id), int(attr_id))] = (int(is_present), int(certainty_id))
        


    #####parts######
    df = pd.read_csv(os.path.join(args.data_dir, 'parts','part_locs.txt'),
                    delimiter=' ',
                    names=['image_id', 'part_id', 'x', 'y', 'visible'])
    # keys = tuple(df['image_id'], df['part_id'])\
    keys = tuple(df[['image_id', 'part_id']].apply(tuple, axis=1))
    values = list(df[['x', 'y', 'visible']].apply(tuple, axis=1))
    part_id_2_part_locs = dict(zip(keys, values))

    return class_2_name, image_id_2_image_path, image_id_2_seg_path, image_id_2_class, image_id_2_bbox, image_id_2_split, attr_id_2_name, class_id_2_attr_makeup, image_id_and_attr_id_2_certainty_id, part_id_2_part_locs

class Part:
    def __init__(self, id:int, x:int, y:int, visible:int):
        self.id = id
        self.name = parts_mapping[id]
        self.x = x
        self.y = y
        self.visible = visible

class Attribute:
    def __init__(self, id:int, name, certainty:int):
        self.id = id
        self.name = name
        self.certainty_id = certainty
        self.certainty = certainties_mapping[certainty]

    def __str__(self):
        return f"{self.id} ({self.name}): {self.certainty_id}-{self.certainty}"

class CUB_Image:
    def __init__(self, id, name, img_path, seg_path, attributes, bbox, label, parts: List[Part]):
        self.id = id
        # self.image = Image.open(img_path)
        # self.segmentation = Image.open(seg_path)
        self.segmentation_threshold = 128 # png is uint8 and seems to be ~45 intensity per labeler
        self.name = name
        self.img_path = img_path
        self.seg_path = seg_path
        self.attributes = attributes
        self.certainty_threshold = 4 # only want visible parts
        self.bbox = tuple(int(x) for x in bbox)
        self.label = label

        self.back = parts[0]
        self.beak = parts[1]
        self.belly = parts[2]
        self.breast = parts[3]
        self.crown = parts[4]
        self.forehead = parts[5]
        self.left_eye = parts[6]
        self.left_leg = parts[7]
        self.left_wing = parts[8]
        self.nape = parts[9]
        self.right_eye = parts[10]
        self.right_leg = parts[11]
        self.right_wing = parts[12]
        self.tail = parts[13]
        self.throat = parts[14]
    
    def attribute(self, id):
        return self.attributes[id] if id in self.attributes else None

    def part(self, part_id):
        if part_id == 1:
            return self.back
        elif part_id == 2:
            return self.beak
        elif part_id == 3:
            return self.belly
        elif part_id == 4:
            return self.breast
        elif part_id == 5:
            return self.crown
        elif part_id == 6:
            return self.forehead
        elif part_id == 7:
            return self.left_eye
        elif part_id == 8:
            return self.left_leg
        elif part_id == 9:
            return self.left_wing
        elif part_id == 10:
            return self.nape
        elif part_id == 11:
            return self.right_eye
        elif part_id == 12:
            return self.right_leg
        elif part_id == 13:
            return self.right_wing
        elif part_id == 14:
            return self.tail
        elif part_id == 15:
            return self.throat
        else:
            raise ValueError(f"Invalid part_id: {part_id}")
    
    def parts(self):
        return [self.back, self.beak, self.belly, self.breast, self.crown, self.forehead, self.left_eye, self.left_leg, self.left_wing, self.nape, self.right_eye, self.right_leg, self.right_wing, self.tail, self.throat]

    def image_as_pil(self):
        return Image.open(self.img_path) #Images are loaded in RGB mode

    def image_as_np(self):
        return np.array(Image.open(self.img_path))

    def show_bbox(self)-> Image.Image:
        x,y,width,height = self.bbox
        np_img = np.array(Image.open(self.img_path))
        return Image.fromarray(cv2.rectangle(np_img, (x, y), (x + width, y + height), (0, 255, 0), 2))
    
    def show_masked_image(self):
        """Show the segmentation mask of the image with a black background."""
        # Convert the image to numpy array
        image_array = np.array(Image.open(self.img_path))
        seg_array = np.array(Image.open(self.seg_path))
        # Create a mask
        mask = np.zeros_like(image_array)
        mask[seg_array >= self.segmentation_threshold] = [0, 0, 0]
        seg_array = np.expand_dims(seg_array, axis=-1)
        seg_array = np.repeat(seg_array, 3, axis=-1)
        overlay = np.where((seg_array >= self.segmentation_threshold), image_array, mask)
        return Image.fromarray(overlay)
    
    def show_mask_on_image(self):
        """Show the segmentation mask on the image with an orange tint."""
        image_array = np.array(Image.open(self.img_path))
        seg_array = np.array(Image.open(self.seg_path))
        
        # Define the orange color
        orange = np.array([255, 165, 0])
        # Create a mask where the condition is met
        mask = seg_array >= self.segmentation_threshold
        
        # Apply the orange tint where the mask is True
        # Convert to float for accurate blending
        image_array = image_array.astype(float)
        overlay = np.zeros_like(image_array)
        overlay[mask] = orange
        
        # Blend the image with the overlay
        alpha = 0.5  # Transparency factor for the overlay
        # image_array = (1 - alpha) * image_array + alpha * overlay
        image_array = image_array + alpha * overlay
        
        # Clip values to the valid range for image display
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        return Image.fromarray(image_array)
        

    def show_parts(self):
        # Convert the image to numpy array
        image_array = np.array(Image.open(self.img_path))
        
        for part in self.parts():
            if part.visible == 1:
                cv2.circle(image_array, (part.x, part.y), 5, (255, 255, 0), -1)
        # return mask
        return Image.fromarray(image_array)

class CUB:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.CUB_train_set:List[CUB_Image] = []
        self.CUB_val_set:List[CUB_Image] = []
        self.attributes_gt ={}

        all_mappings = load_data(args)
        class_2_name = all_mappings[0]
        image_id_2_image_path = all_mappings[1]
        image_id_2_seg_path = all_mappings[2]
        image_id_2_class = all_mappings[3]
        image_id_2_bbox = all_mappings[4]
        image_id_2_split = all_mappings[5]
        attr_id_2_name = all_mappings[6]
        class_id_2_attr_makeup = all_mappings[7]
        image_id_and_attr_id_2_certainty_id = all_mappings[8]
        part_id_2_part_locs = all_mappings[9]

        self.classes = list(class_2_name.keys())


        for class_id, attributes in class_id_2_attr_makeup.items():
            self.attributes_gt[class_id] = [attr/100 for attr in attributes]

        for i in range(len(image_id_2_image_path.keys())):
            image_id = i + 1
            path = os.path.join(self.data_dir, 'images', image_id_2_image_path[image_id])
            seg_path = os.path.join(self.data_dir, 'segmentations', image_id_2_seg_path[image_id])
            class_id = image_id_2_class[image_id]
            class_name = class_2_name[class_id]
            bbox = image_id_2_bbox[image_id]
            split = image_id_2_split[image_id]
            attributes = {}
            for j in range(1,313):
                is_present, certainty = image_id_and_attr_id_2_certainty_id[(image_id, j)]
                if is_present and certainty >= args.certainty_threshold:
                    attributes[j] = Attribute(j, attr_id_2_name[j], certainty)

            parts = []
            for i in range(1,16):
                x,y,visible = part_id_2_part_locs[(image_id,i)]
                parts.append(Part(int(i), int(x), int(y), int(visible)))


            label = class_id

            if split == 1:
                self.CUB_train_set.append(CUB_Image(image_id, class_name, path, seg_path, attributes, bbox, label, parts))
            else:
                self.CUB_val_set.append(CUB_Image(image_id, class_name, path, seg_path, attributes, bbox, label, parts))

    def __getitem__(self, idx):
        return self.CUB_images[idx]

    def __len__(self):
        return len(self.CUB_images)


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
        target = tv_tensors.Mask(torch.cat([background_mask, class1], dim=0))
        img, target = self.transform(img, target)

        if target.shape != (2, 224, 224):
            print('error at',self.image_ids[idx])
            print('target[masks].shape',target['masks'].shape)
            print('mask.shape',mask.shape)
            print('self.image_paths[idx]',self.image_paths[idx])
            print('self.mask_paths[idx]',self.mask_paths[idx])
        return img, target
    
