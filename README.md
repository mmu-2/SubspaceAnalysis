# SubspaceAnalysis


## Getting Started

```
conda create -n "cub" python==3.12
conda activate cub
pip install -r requirements.txt
```

You can run classification code with the following lines:
```
python classification_pipeline.py --epochs 200 \
    --k 1024 \
    --data_dir ./fashionmnist/ \
    --dataset fmnist \
    --model rn18 \
    --projection \
    --no_log
```

## Datasets


### CUB_200_2011



0. I later found a more informative README with dataset information here: https://drive.google.com/drive/folders/1kFzIqZL_pEBVR7Ca_8IKibfWoeZc3GT1

1. Download dataset and segmentations from here: https://www.vision.caltech.edu/datasets/cub_200_2011/
2. A separate paper provides the 10 sentence descriptions here: https://github.com/reedscot/cvpr2016?tab=readme-ov-file

2. I currently expect a hierarchy like this:
```
CUB_200_2011
│   attributes
    -certainties.txt
    -class_attributes_labels_continuous.txt
    -image_attribute_labels.txt
|   images
    |   001.Black_footed_Albatross
        -Black_Footed_Albatross_0001_796111.jpg
        -...
    |   ...
    |   200.Common_Yellowthroat
|   parts
    -part_click_locs.txt
    -part_locs.txt
    -parts.txt
|   segmentations (you'll need to move this here)
    |   001.Black_footed_Albatross
        -Black_Footed_Albatross_0001_796111.png
        -...

-attributes.txt
-bounding_boxes.txt
-classes.txt
-image_class_labels.txt
-images.txt
-README.md
-train_test_split.txt
```

- certainties: 1 not visible; 2 guessing; 3 probably; 4 definitely
- class_attributes_labels_continuous.txt: 200 rows: 312 attributes per row as percentage of presence of attribute
- image_attribute_labels.txt: 3,677,856 lines = 312 * 11,788. <image_id> <attribute_id> <is_present> <certainty_id> <time>
- part_click_locs: <image_id> <part_id> <x> <y> <visible> <time>
- part_locs: <image_id> <part_id> <x> <y> <visible>
This differs from locs because it combines the MTurk data by finding the median of values.

- parts: 1-back;2-beak;3-belly;4-breast;5-crown;6-forehead;7-left eye;8-left leg;9-left wing;10-nape;11-right eye
12 right-leg;13-right wing;14-tail;15-throat

- attributes: maps number to attribute
- bounding_boxes: 1 bounding box per image <image_id> <x> <y> <width> <height>
- classes: maps class id to class name
- image_class_labels: maps image id to a class
- images: image id to image path
- train_test_split: maps each image id to 1-train;0-test split.

- segmentations: grayscale images with ~4 votes on the segmentation. Most are format 'L' (8-bit grayscale), but some are 'LA' (L with alpha)


### CelebA

There are a couple labels provided:
- The dataset has 40 attributes with boolean flags to show if they exist in an image.
- Identity id. There are ids from 1 to 10,177 with 202,599 samples which is approx 19.9 images per person.
- Bounding boxes for the face. (x_1 y_1 width height)
- Landmark pixel locations of eyes, nose, left/right side of mouth

I've coded up the ability to run classification by identity, but since then, I've realized that the validation/test sets 
are purposely different celebrities, so there's no overlap whatsoever and classification will always have accuracy of 0 on the val/test sets.



### Caltech
There's two version, but 101 has mask annotations and 256 does not, so I only describe that Caltech101.


Labels:
- annotation (contours): shape is (2, N variable points) - First axis is x,y but we should transpose it for our use.
- category (class)

The Pytorch torchvision code is bugged: https://github.com/pytorch/vision/issues/7748.
If you scroll down a bit, you can also see the layout of the code.

If I want to do bounding boxes or masks, I can steal working code from the unmerged fix branch https://github.com/pytorch/vision/pull/7752
