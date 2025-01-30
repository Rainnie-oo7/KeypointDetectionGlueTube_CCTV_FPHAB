import os.path as osp
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import numpy as np
import sys
import os
from pathlib import Path
from os.path import isfile, join
from os import listdir
import re
import cv2
import json
import PIL.Image
from collections import Counter
import transforms
from torch.utils.data import Dataset, DataLoader
# from torchvision import tv_tensors


# sys.path.append('/home/boris.grillborzer/PycharmProjects/PoseEstimationIRD')


class SkeletonDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo  # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            # All objects are glue tubes
            bboxes_labels_original = ['Glue tube' for _ in bboxes_original]

        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]

            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                         bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']

            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1, 2, 2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):  # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj):  # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

            # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)  # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original],
                                                    dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                    bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)    #12057 #216 Dir/Annots mit => 12057 Zeilen

def chain_imagepaths(folder):
    image_paths = []
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".png"):
            image_paths.append(os.path.join(folder, file)) #Append the image-Paths to the list
    return image_paths

def getimagefilename(folder):
    image_filenames = []
    files = sorted(os.listdir(folder))
    for file in files:
        if file.endswith(".png"):
            image_filenames.append(file) #Append the image-filename to the list
    return image_filenames

def chain_txtpaths(path):
    txtfile_paths = []
    for txtfile in sorted(os.listdir(path)):
        filepath = os.path.join(path_skelraw_data, txtfile)
        txtfile_paths.append(filepath)
    return txtfile_paths


def getabstracted_txtpaths(txtfile_paths):
    abstracted_txtfile_names = []
    for txtpath in txtfile_paths:

        abstracted_txtfile_names.append(Path(txtpath).stem)
    return abstracted_txtfile_names

def get_movementrepetition(abstracted_txtfile_names):
    repetition_infos = []
    for txtfile in abstracted_txtfile_names:
        txtbenennungsids = txtfile.split("_")
        repetition_info = txtbenennungsids[3]   #Vierte Stelle ist Repetition Number
        repetition_infos.append(repetition_info)
    return repetition_infos

def make_two_digit_txtpaths(file_path):
    # Extrahiere den Dateinamen
    file_name = os.path.basename(file_path)
    # Ersetze die Zahl hinter dem dritten Unterstrich durch eine zweistellige Zahl
    updated_file_name = re.sub(
        r'(^\d+_\d+_\d+_)(\d+)(?=_)',  # Der Teil, der die Zahl nach dem dritten Unterstrich findet
        lambda x: f"{x.group(1)}{int(x.group(2)):02}",  # Setze die Zahl auf zweistellig
        file_name
    )
    # Erstelle den neuen Pfad mit dem angepassten Dateinamen
    return os.path.join(os.path.dirname(file_path), updated_file_name)


def recreate_txtpaths_sorted(sorted_txtpaths_with_twodigit):
    corrected_file_paths = []
    for p in sorted_txtpaths_with_twodigit:
        # Extrahiere den Dateinamen
        file_name = os.path.basename(p)

        # Zerlege den Dateinamen nach Unterstrichen und entferne führende Nullen
        parts = file_name.split('_')
        if len(parts) > 3:
            # Teile die Zahl nach dem dritten Unterstrich
            parts[3] = str(int(parts[3]))  # Entfernt führende Null, wenn vorhanden

        # Setze den Dateinamen wieder zusammen
        updated_file_name = '_'.join(parts)

        # Erstelle den neuen Pfad mit dem angepassten Dateinamen
        corrected_file_paths.append(os.path.join(path_skelraw_data, updated_file_name))
    return corrected_file_paths

def recreate_imgspaths_sorted(path_dataset_depthimages_two_digits):
    corr_imgspaths = []
    for p in path_dataset_depthimages_two_digits:
        # Extrahiere den Dateinamen
        file_name = os.path.basename(p)

        # Zerlege den Dateinamen nach Unterstrichen und entferne führende Nullen
        parts = file_name.split('_')
        if len(parts) > 3:
            # Teile die Zahl nach dem dritten Unterstrich
            parts[3] = str(int(parts[3]))  # Entfernt führende Null, wenn vorhanden

        # Setze den Dateinamen wieder zusammen
        updated_file_name = '_'.join(parts)
        corr_imgspaths.append(os.path.join(path_dataset_depthimages, updated_file_name))
        # Erstelle den neuen Pfad mit dem angepassten Dateinamen
    return corr_imgspaths


def create_imgpaths(corr_imgdirpaths):
    corr_imgs_paths = []
    for dir in corr_imgdirpaths:
        full_dir_path = dir
        if os.path.isdir(full_dir_path):
            files = sorted(os.listdir(full_dir_path))
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(full_dir_path, file)
                    corr_imgs_paths.append(img_path)

    return corr_imgs_paths


def load_skeleton(txtfile_paths):
    skeleton_list2d = []
    frames_nrs = []
    counter = 0
    # Erste 3 Columns unwichtig \\ ignoren.
    line_length = 3 + 25 * 7

    for txtpath in txtfile_paths:

        with open(txtpath) as f:
            for line in f.readlines():
                line = line.strip()
                if "Version" in line:
                    continue
                words = line.split(",")
                frame_nr = int(words[0])    #Erstes Column
                joints2d = {}
                # print(line)
                # print(len(words), line_length)
                if len(words) != line_length:
                    continue

                for idx in range(25):
                    joint_name = words[3 + idx * 7].strip('(')
                    # joint_tracked = words[3 + idx * 7 + 1]

                    # joint_x = float(words[3 + idx * 7 + 2])
                    # joint_y = float(words[3 + idx * 7 + 3])
                    # joint_z = float(words[3 + idx * 7 + 4])

                    joint_dx = float(words[3 + idx * 7 + 5])
                    joint_dy = float(words[3 + idx * 7 + 6].strip(')'))
                    # if joint_x == 0 and joint_y == 0 and joint_z == 0:
                    #     joint_dx = 0
                    #     joint_dy = 0
                    # else:
                    #     joint_dx = float(words[3 + idx * 7 + 5])
                    #     joint_dy = float(words[3 + idx * 7 + 6].strip(')'))
                    # joints5d[joint_name] = [joint_x, joint_y, joint_z, joint_dx, joint_dy]
                    # joints3d[joint_name] = [joint_x, joint_y, joint_z]
                    joints2d[joint_name] = [joint_dx, joint_dy]

                skeleton_list2d.append(joints2d)
                frames_nrs.append(frame_nr)
                counter += 1

    return skeleton_list2d, frames_nrs, counter

def getkeypoints_from_dict_out(keypoints_dicted):
    #"keypoints": [[[1019, 487, 1], [1432, 404, 1]], [[861, 534, 1], [392, 666, 1]]]
    # Liste von Listen # Liste von Listen mit Einsen
    list_of_lists = list(keypoints_dicted.values())
    list_of_lists_withone = [values + [1] for values in keypoints_dicted.values()]

    # Flache Liste
    flat_list = [value for sublist in list_of_lists for value in sublist]

    return list_of_lists_withone, list_of_lists, flat_list

def check_framenr_with_skeletonload(frames_nr, image_filename):
    if frames_nr == Path(image_filename).stem is True:
        return True
#
# def count_skeletwentyfivers(skeleton_list3d):
#     #loop to count amount twentyfivers
#     #return count
#     pass

def remove_leading_zero_repetionslist(repetition_infos_with_twodigit):
    cleaned_numbers = [str(int(num)) for num in repetition_infos_with_twodigit]
    return cleaned_numbers

def check_amount_skeletwentyfivers_with_repetitionnr(count, repetition_info):
    #if %25 = 0 continue
    #else print error

    if count == repetition_info is True:
        return True
    pass

# def assign_repetition_to_pic(self):   #Jedes Frame hat eine Annotation
#     self.data_sk, self.txtfile_paths, self.abstracted_txtfile_names, self.repetition_info, self.frames_nr = process_files(self, path_skelraw_data)







