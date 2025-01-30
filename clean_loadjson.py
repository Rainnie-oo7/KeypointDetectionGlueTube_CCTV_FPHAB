import json

# Pfad zur JSON-Datei
file_path = "/home/boris.grillborzer/PycharmProjects/Realtime_Multi-Person_Pose_Estimation/training/dataset/COCO/annotations/person_keypoints_train2014.json"

# JSON-Datei laden
with open(file_path, "r") as file:
    data = json.load(file)

# # Alle Top-Level-Keys der JSON-Daten anzeigen
# print("Top-Level Keys in der JSON-Datei:")
# for key in data.keys():
#     print(f"- {key}")

# # edge_index
# if "categories" in data:
#     print("\nKategorien:")
#     categories = data["categories"]
#     for category in categories:
#         print(category)
# else:
#     print("\nKeine Kategorien in der JSON-Datei gefunden.")

# Annotationnen
keypoints = []
if "categories" in data:
    print("\nKategorien:")
    annotations = data["annotations"]

    for i,_ in enumerate(annotations):
        a = data["annotations"][i]["keypoints"]
        keypoints.append(a)
        print(keypoints)
    else:
        print("\nKeine Kategorien in der JSON-Datei gefunden.")

print("yokis")
#0: Der Keypoint ist nicht vorhanden (fehlt oder nicht annotiert).
#1: Der Keypoint ist vorhanden, aber nicht sichtbar (z. B. verdeckt).
#2: Der Keypoint ist vorhanden und sichtbar.

#"info"             dict_keys(['description', 'url', 'version', 'year', 'contributor', 'date_created'])
#["images"][0]      dict_keys(['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id'])
#["licenses"][0]    dict_keys(['url', 'id', 'name'])
#["annotations"][0] dict_keys(['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id'])
#["categories"][0]  dict_keys(['supercategory', 'name', 'skeleton', 'keypoints', 'id'])
#           'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
#           'keypoints': ['nose',
#           'left_eye',
#           'right_eye',
#           'left_ear',
#           'right_ear',
#           'left_shoulder',
#           'right_shoulder',
#           'left_elbow',
#           'right_elbow',
#           'left_wrist',
#           'right_wrist',
#           'left_hip',
#           'right_hip',
#           'left_knee',
#           'right_knee',
#           'left_ankle',
#           'right_ankle']
