

from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.yolo_nas_pose_variants import YoloNASPose_S
from tlc.core import Table
import tlc

checkpoint_path = "C:/Project/super-gradients/notebooks/checkpoints/training-animalpose-yolo-nas-pose-3lc-1/RUN_20250911_163529_443788/ckpt_best.pth"
best_model: YoloNASPose_S = models.get(Models.YOLO_NAS_POSE_S, num_classes=20, checkpoint_path=checkpoint_path).cuda()


train_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/train")
val_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/val")

def metrics_fn(batch, predictor_output):
    return {"keypoints_2d_predicted": [0] * len(batch)}

predictor = tlc.Predictor(best_model, call_fn="predict", disable_preprocess=True)

def collate_fn(batch):
    return [sample["image"] for sample in batch]

tlc.collect_metrics(
    train_table, 
    metrics_fn,
    predictor,
    split="train",
    dataloader_args={"batch_size": 16, "collate_fn": collate_fn},
)

assert True