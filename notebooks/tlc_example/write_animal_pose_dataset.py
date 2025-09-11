from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from tlc.core.builtins.schemas import Keypoints2DSchema
from tlc.core.writers import TableWriter
from tlc.core import Schema, ImageUrlStringValue, Table



def load_table_rows(data, image_root):
    annotations = data["annotations"]  # list of dict
    images = data["images"]  # dict str: str

    rows = []
    for image_id, image_path in tqdm(images.items(), total=len(images), desc="Loading annotations"):
        image_id = int(image_id)
        image_path = Path(image_root) / image_path
    
        if not image_path.exists():
            print(f"Image {image_path} does not exist")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        anns = [a for a in annotations if a["image_id"] == image_id]

        keypoints = {
            "x_max": width,
            "y_max": height,
            "instances": [],
            "instances_additional_data": {
                "label": [],
            },
        }

        for ann in anns:
            kpts = np.array(ann["keypoints"])[:,:2].reshape(-1).tolist()
            visibilities = np.array(ann["keypoints"])[:,2].tolist()
            bb = {"x_min": ann["bbox"][0], "y_min": ann["bbox"][1], "x_max": ann["bbox"][2], "y_max": ann["bbox"][3]}
            label = ann["category_id"]

            keypoints["instances"].append({
                "vertices_2d": kpts,
                "vertices_2d_additional_data": {
                    "visibilities": visibilities,
                },
                "bbs_2d": [bb],
            })
            keypoints["instances_additional_data"]["label"].append(label)
    
        rows.append(
            {
                "image": image_path.as_posix(),
                "keypoints_2d": keypoints,
            }
        )

    return rows

def write_table(rows, keypoint_names, cats, num_keypoints, skeleton):

    EDGE_COLORS = list(map(
        lambda x: f"#{'{:02x}'.format(x[0])}{'{:02x}'.format(x[1])}{'{:02x}'.format(x[2])}", 
        [
            [127, 0, 255],
            [91, 56, 253],
            [55, 109, 248],
            [19, 157, 241],
            [18, 199, 229],
            [54, 229, 215],
            [90, 248, 199],
            [128, 254, 179],
            [164, 248, 158],
            [200, 229, 135],
            [236, 199, 110],
            [255, 157, 83],
            [255, 109, 56],
            [255, 56, 28],
            [255, 0, 0],
        ]
    ))

    KEYPOINT_COLORS = list(map(
        lambda x: f"#{'{:02x}'.format(x[0])}{'{:02x}'.format(x[1])}{'{:02x}'.format(x[2])}", 
        [
            [148, 0, 211],
            [75, 0, 130],
            [0, 0, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 165, 0],
            [255, 69, 0],
            [255, 0, 0],
            [139, 0, 0],
            [128, 0, 128],
            [238, 130, 238],
            [186, 85, 211],
            [148, 0, 211],
            [0, 255, 255],
            [0, 128, 128],
            [0, 0, 139],
            [0, 0, 255],
            [0, 255, 0],
            [255, 69, 0],
            [255, 127, 14],
        ],
    ))

    table_writer = TableWriter(
        table_name="initial",
        dataset_name="animalpose",
        project_name="animalpose",
        column_schemas={
            "image": Schema(value=ImageUrlStringValue()),
            "keypoints_2d": Keypoints2DSchema(
                classes=cats,
                num_keypoints=num_keypoints,
                lines=skeleton,
                line_attributes=[{"internal_name": f"edge_{i}", "display_color": EDGE_COLORS[i]} for i in range(len(skeleton) // 2)],
                point_attributes=[{"internal_name": kpt_name, "display_color": KEYPOINT_COLORS[i]} for i, kpt_name in enumerate(keypoint_names)],
                include_per_point_visibilities=True,
                flip_indices=[1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19],
            ),
        }
    )

    for row in tqdm(rows, total=len(rows), desc="Writing table"):
        table_writer.add_row(row)
    
    table = table_writer.finalize()
    return table

def tlc_table_from_super_gradients(annotations_file: str, image_root: str) -> Table:
    data = json.load(open(annotations_file))
    rows = load_table_rows(data, image_root)

    keypoint_names = data["categories"][0]["keypoints"]
    cats = {cat["id"]: cat["name"] for cat in data["categories"]}
    num_keypoints = len(keypoint_names)
    skeleton = np.array(data["categories"][0]["skeleton"]).reshape(-1).tolist()

    table = write_table(rows, keypoint_names, cats, num_keypoints, skeleton)
    return table

if __name__ == "__main__":
    annotations_file = "D:/Data/animalpose/keypoints.json"
    image_root = "D:/Data/animalpose/images"
    table = tlc_table_from_super_gradients(annotations_file, image_root)
