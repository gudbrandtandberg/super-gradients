from __future__ import annotations
from tlc.core import Table
import numpy as np

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset
from tlc.core.builtins.types import KeypointHelper
import cv2
import numpy as np

def hex_to_rgb(hex_color):
    """hex_color is a string like #RRGGBB"""
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def rgb_to_hex(rgb_color):
    """rgb_color is a tuple like (255, 255, 255)"""
    return "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])

class TLCPoseEstimationDataset(AbstractPoseEstimationDataset):
    """
    Dataset class for training pose estimation models on Animal Pose dataset.
    """

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        table: Table,
        transforms: list[AbstractKeypointTransform],
    ):
        """

        """
        self.table = table

        keypoint_attributes = KeypointHelper.get_keypoint_attributes_from_table(table, "keypoints_2d")
        keypoint_colors = [hex_to_rgb(keypoint_attribute["display_color"]) for keypoint_attribute in keypoint_attributes]
        keypoint_names = [keypoint_attribute["internal_name"] for keypoint_attribute in keypoint_attributes]
        lines_attributes = KeypointHelper.get_line_attributes_from_table(table, "keypoints_2d")
        edge_colors = [hex_to_rgb(line_attribute["display_color"]) for line_attribute in lines_attributes]
        skeleton = np.array(table.rows_schema["keypoints_2d"]["instances"]["lines"].default_value).reshape(-1, 2).tolist()

        super().__init__(
            transforms=transforms,
            num_joints=len(keypoint_names),
            edge_links=skeleton,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

    def __len__(self):
        return len(self.table)

    def load_sample(self, index) -> PoseEstimationSample:
        """
        :param image:              Associated image with a sample. Can be in [H,W,C] or [C,H,W] format
        :param mask:               Target mask in [H,W] format
        :param joints:             Target joints in [NumInstances, NumJoints, 3] format.
                                Last dimension contains (x,y,visibility) for each joint.
        :param areas:              (Optional) Numpy array of [N] shape with area of each instance.
                                Note this is not a bbox area, but area of the object itself.
                                One may use a heuristic `0.53 * box area` as object area approximation if this is not provided.
        :param bboxes_xywh:        (Optional) Numpy array of [N,4] shape with bounding box of each instance (XYWH)
        :param additional_samples: (Optional) List of additional samples for the same image.
        :param is_crowd:           (Optional) Numpy array of [N] shape with is_crowd flag for each instance
        """
        file_path = self.table[index]["image"]
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        mask = np.ones(image.shape[:2], dtype=np.float32)

        keypoints = []
        bboxes = []

        for instance in self.table[index]["keypoints_2d"]["instances"]:
            points = np.array(instance["vertices_2d"]).reshape(-1, 2)
            visibilities = np.array(instance["vertices_2d_additional_data"]["visibilities"]).reshape(-1, 1)
            x_min = instance["bbs_2d"][0]["x_min"]
            y_min = instance["bbs_2d"][0]["y_min"]
            x_max = instance["bbs_2d"][0]["x_max"]
            y_max = instance["bbs_2d"][0]["y_max"]
            box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
            keypoints.append(np.concatenate([points, visibilities], axis=-1))
            bboxes.append(box)

        gt_joints = np.array(keypoints)
        gt_bboxes = np.array(bboxes)
        gt_iscrowd = np.array([0] * len(keypoints), dtype=bool)
        gt_areas = np.array([box[2] * box[3] * 0.53 for box in gt_bboxes], dtype=np.float32)

        return PoseEstimationSample(
            image=image,
            mask=mask,
            joints=gt_joints,
            areas=gt_areas,
            bboxes_xywh=gt_bboxes,
            is_crowd=gt_iscrowd,
            additional_samples=None,
        )

if __name__ == "__main__":
    import time
    import tqdm

    table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/initial_0002")
    dataset = TLCPoseEstimationDataset(table, [])
    

    start = time.time()

    for i in tqdm.tqdm(range(len(dataset)), total=len(dataset), desc="Loading dataset"):
        dataset[i]

    end = time.time()
    print(f"Time taken: {end - start} seconds")
