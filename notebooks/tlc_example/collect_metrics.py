

from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.yolo_nas_pose_variants import YoloNASPose_S
from tlc.core import Table
from pose_metrics_collector import SuperGradientsPoseMetricsCollector
import tlc
import time

checkpoint_path = "C:/Project/super-gradients/notebooks/checkpoints/training-animalpose-yolo-nas-pose-3lc-1/RUN_20250911_163529_443788/ckpt_best.pth"
best_model: YoloNASPose_S = models.get(Models.YOLO_NAS_POSE_S, num_classes=20, checkpoint_path=checkpoint_path).cuda()

def collate_fn(batch):  # Note: needs to be defined at module level for worker processes to be able to access it
    return [sample["image"] for sample in batch]

if __name__ == "__main__":

    tlc.init(project_name="animalpose")

    train_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/train")
    val_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/val")

    predictor = tlc.Predictor(best_model, call_fn="predict", disable_preprocess=True)


    start = time.time()
    mc_batch_size = 32
    mc_num_workers = 8

    tlc.collect_metrics(
        val_table, 
        SuperGradientsPoseMetricsCollector(train_table),
        predictor,
        split="train",
        dataloader_args={"batch_size": mc_batch_size, "collate_fn": collate_fn, "num_workers": mc_num_workers, "persistent_workers": mc_num_workers > 0},
        collect_aggregates=False,
    )
    end = time.time()
    print(f"Collected metrics for train set in {end - start} seconds (batch size {mc_batch_size}, workers {mc_num_workers})")

    # Unfair comparisons, but still interesting. Worker spawn time is dominating, but would not be the case for larger datasets or subsequent calls (due to persistent workers).
    # Collected metrics for train set in 98.3032398223877 seconds (batch size 4, workers 0)
    # Collected metrics for train set in 82.96371078491211 seconds (batch size 8, workers 0)
    # Collected metrics for train set in 44.35213351249695 seconds (batch size 16, workers 0)
    # Collected metrics for train set in 40.906614780426025 seconds (batch size 32, workers 0)
    # Collected metrics for train set in 37.041160106658936 seconds (batch size 36, workers 0)
    # Collected metrics for train set in 128.1499412059784 seconds (batch size 4, workers 4)
    # Collected metrics for train set in 89.74156379699707 seconds (batch size 8, workers 4)
    # Collected metrics for train set in 119.00389504432678 seconds (batch size 8, workers 8)
    # Collected metrics for train set in 100.95760107040405 seconds (batch size 32, workers 8)