

from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.yolo_nas_pose_variants import YoloNASPose_S
from tlc.core import Table
from pose_metrics_collector import SuperGradientsPoseMetricsCollector
import tlc

checkpoint_path = "C:/Project/super-gradients/notebooks/checkpoints/training-animalpose-yolo-nas-pose-3lc-1/RUN_20250911_163529_443788/ckpt_best.pth"
best_model: YoloNASPose_S = models.get(Models.YOLO_NAS_POSE_S, num_classes=20, checkpoint_path=checkpoint_path).cuda()

if __name__ == "__main__":

    tlc.init(project_name="animalpose")
    
    train_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/train")
    val_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/val")

    predictor = tlc.Predictor(best_model, call_fn="predict", disable_preprocess=True)

    def collate_fn(batch):
        return [sample["image"] for sample in batch]

    tlc.collect_metrics(
        val_table, 
        SuperGradientsPoseMetricsCollector(train_table),
        predictor,
        split="train",
        dataloader_args={"batch_size": 4, "collate_fn": collate_fn},
        collect_aggregates=False,
    )

    tlc.collect_metrics(
        train_table, 
        SuperGradientsPoseMetricsCollector(val_table),
        predictor,
        split="val",
        dataloader_args={"batch_size": 4, "collate_fn": collate_fn},
        collect_aggregates=False,
    )
