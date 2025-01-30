import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import numpy as np
from configs import config_seamer as cf
from distinctipy import distinctipy
from Mydataset import SkeletonDataset
from gluetube_bobtest import ClassDataset
import albumentations as A # Library for augmentations

def train_transform():          #For ""Augmentations"
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


if __name__ == '__main__':
    root = "/home/boris.grillborzer/PycharmProjects/ThirdOrial/41KeypointDetectionUhuKleber/glue_tubes_keypoints_dataset_134imgs/train"

    # Lade Pfade aus der Konfiguration
    # path_skelraw_data = cf.path_skelraw_data
    # path_dataset_depthimages = cf.path_dataset_depthimages
    dataset = ClassDataset(root, transform=train_transform())
    idx = 17

    # image, data, labels, sk_path = dataset[idx]
    img, target = dataset[idx]


    print("yochen")
    print("yochen")
    print("yochen")
