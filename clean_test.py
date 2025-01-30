import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import numpy as np
from configs import config_seamer as cf
from distinctipy import distinctipy
from Mydataset import SkeletonDataset
import albumentations as A # Library for augmentations

def train_transform():          #For ""Augmentations"
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


if __name__ == '__main__':
    path_skelraw_data = cf.path_skelraw_data
    path_dataset_depthimages = cf.path_dataset_depthimages

    dataset = SkeletonDataset(path_skelraw_data, path_dataset_depthimages, transform=train_transform(), demo=False)
    idx = 17

    # image, data, labels, sk_path = dataset[idx]
    img, target = dataset[idx]
    print("yochen")
    print("yochen")
    print("yochen")

    #Skel u. Bild Anzeigen
    #Preliminaries
    keypoints = []

    for values in data.values():
        keypoints.append(values)

    # Convert the list into a NumPy array for easier manipulation
    keypoints_array = np.array(keypoints)

    # Extract x, y, z coordinates
    x = keypoints_array[:, 0]  # All rows, first column (x)
    y = keypoints_array[:, 1]  # All rows, second column (y)
    # z = keypoints_array[:, 2]  # All rows, third column (z)

    # for i in range(25):
    #     x = keypoints[i][0]             #x = [kp[0] for kp in keypoints]
    #     y = keypoints[i][1]             #y = [kp[1] for kp in keypoints]
    #     z = keypoints[i][2]             #z = [kp[2] for kp in keypoints]

    #Farben
    colors = distinctipy.get_colors(len(labels))
    # distinctipy.color_swatch(colors) # visualize c-map

    img = mpimg.imread(image)  # Replace with your image file

    # Create a figure for the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot
    ax.scatter(x, y, z, c='r', marker='o')

    # Overlay the image at each keypoint (this works for 2D images; for 3D you would need custom markers)
    # ax.imshow(img, extent=[min(y), max(y), min(z), max(z)], aspect='auto', alpha=0.5, zorder=-1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Label: {idx}, Img-Pfad: {image} \n             Skel.-Pfad: {sk_path}")

    # Show the plot
    plt.show()

    print("SS")
    print("SS")
    print("SS")
