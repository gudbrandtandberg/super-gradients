from super_gradients.training.utils.predict.predictions import PoseEstimationPrediction
from super_gradients.training.utils.predict.prediction_pose_estimation_results import ImagesPoseEstimationPrediction, ImagePoseEstimationPrediction
from tlc.core.builtins.schemas import Keypoints2DSchema
from tlc.core.builtins.types.geometry_helper import GeometryHelper
import tlc
from typing import Any


class SuperGradientsPoseMetricsCollector(tlc.MetricsCollector):
    def __init__(self, input_table: tlc.Table):
        self.table = input_table
        super().__init__(compute_aggregates=False)

    def compute_metrics(self, batch: Any, predictor_output: tlc.PredictorOutput):
        predictions = predictor_output.forward
        assert isinstance(predictions, ImagesPoseEstimationPrediction)
        keypoints_2d_predicted = []

        for prediction in predictions:
            assert isinstance(prediction, ImagePoseEstimationPrediction)
            h, w, _ = prediction.image.shape
            predicted_row = {
                "x_max": w,
                "y_max": h,
                "instances": [],
                "instances_additional_data": {
                    "label": [],
                    "confidence": [],
                },
            }
            
            assert isinstance(prediction.prediction, PoseEstimationPrediction)
            for poses, boxes, scores in zip(prediction.prediction.poses, prediction.prediction.bboxes_xyxy, prediction.prediction.scores):
                xys = poses[:, :2].reshape(-1).tolist()
                visibilities = poses[:, 2].reshape(-1).tolist()
                predicted_row["instances"].append({
                    "vertices_2d": xys,
                    "vertices_2d_additional_data": {
                        "confidence": visibilities,
                    },
                    "bbs_2d": [{"x_min": boxes[0], "y_min": boxes[1], "x_max": boxes[2], "y_max": boxes[3]}],
                })
                predicted_row["instances_additional_data"]["label"].append(-1)
                predicted_row["instances_additional_data"]["confidence"].append(scores)

            keypoints_2d_predicted.append(predicted_row)  # poses, scores, image_shape, bboxes_xyxy

        return {"keypoints_2d_predicted": keypoints_2d_predicted}


    @property
    def column_schemas(self):
        return {
            "keypoints_2d_predicted": Keypoints2DSchema(
                classes=self.table.get_value_map("keypoints_2d.instances_additional_data.label"),
                num_keypoints=20,
                include_per_point_confidences=True,
                include_per_object_confidences=True,
                lines=GeometryHelper.get_lines_from_table(self.table, "keypoints_2d"),
                line_attributes=GeometryHelper.get_line_attributes_from_table(self.table, "keypoints_2d"),
                point_attributes=GeometryHelper.get_keypoint_attributes_from_table(self.table, "keypoints_2d"),
            )
        }
