import json
import os
import re

def extract_annotations(json_path, output_dir):
    # JSON-Datei laden
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    # Mapping von Bild-ID zu Dateinamen
    image_id_to_filename = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # Gruppierung der Annotationen nach Bild-ID
    annotations_by_image = {}
    for ann in data.get("annotations", []):
        image_id = ann["id"]    # Oder besser image_id
        bbox = ann.get("bbox", [])
        keypoints = ann.get("keypoints", [])
        corrected_bbox = subtracted_correct_bbox(bbox) #Does make x, widht, y height Coco Format instead of x, y, width, height u dig
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = {"bboxes": [], "keypoints": []}

        annotations_by_image[image_id]["bboxes"].append(corrected_bbox)
        annotations_by_image[image_id]["keypoints"].append(keypoints)

    # JSON-Dateien für jedes Bild speichern
    for image_id, annotations in annotations_by_image.items():
        filename = image_id_to_filename.get(image_id, f"CCTV_{image_id}.png")
        json_filename = os.path.join(output_dir, f"{filename.replace('.png', '.json')}")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=4)

    print(f"Extraktion abgeschlossen. JSON-Dateien gespeichert in: {output_dir}")

def make_two_digit_txtpaths(file_path):
    # Extrahiere den Dateinamen
    file_name = os.path.basename(file_path)
    # Ersetze die Zahl hinter dem ersten Unterstrich durch eine zweistellige Zahl
    updated_file_name = re.sub(
        r'(^\d+_)(\d+)(?=_)',  # Der Teil, der die Zahl nach dem dritten Unterstrich findet
        lambda x: f"{x.group(1)}{int(x.group(2)):02}",  # Setze die Zahl auf zweistellig
        file_name
    )
    # Erstelle den neuen Pfad mit dem angepassten Dateinamen
    # return os.path.join(os.path.dirname(file_path), updated_file_name)
    return file_name


def recreate_txtpaths_sorted(sorted_txtpaths_with_twodigit):
    corrected_file_paths = []
    for p in sorted_txtpaths_with_twodigit:
        # Extrahiere den Dateinamen
        file_name = os.path.basename(p)

        # Zerlege den Dateinamen nach Unterstrichen und entferne führende Nullen
        parts = file_name.split('_')
        if len(parts) > 3:
            # Teile die Zahl nach dem dritten Unterstrich
            parts[1] = str(int(parts[3]))  # Entfernt führende Null, wenn vorhanden

        # Setze den Dateinamen wieder zusammen
        updated_file_name = '_'.join(parts)

        # Erstelle den neuen Pfad mit dem angepassten Dateinamen
        corrected_file_paths.append(os.path.join(path_skelraw_data, updated_file_name))
    return corrected_file_paths


# def subtracted_correct_bbox(bboxes):
#     corrected_bboxes = []
#     for bbox in bboxes:
#         x = bbox[0]
#         y = bbox[1]
#         width = bbox[2]
#         height = bbox[3]
#
#         corrected_bboxes.append([x, width, y, height])
#
#     return corrected_bboxes

def subtracted_correct_bbox(bboxes):
    corrected_bboxes = []
    for bbox in bboxes:
        x = bbox[0]
        y = bbox[2]
        width = bbox[1]
        height = bbox[3]

        corrected_bboxes.append([x, width, y, height])

    return corrected_bboxes


# Beispielaufruf
extract_annotations("/home/boris.grillborzer/PycharmProjects/KeypointDetectionCCTV/archivetest/CCTVAnnotations.json", "/home/boris.grillborzer/PycharmProjects/KeypointDetectionCCTV/train/annotations")
