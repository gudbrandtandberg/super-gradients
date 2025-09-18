"""For debugging pose metrics collection"""
import time

from super_gradients.training import models
from super_gradients.common.object_names import Models
from tlc.core import Table
import tlc
from pose_metrics_collector import SuperGradientsPoseMetricsCollector

table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/val")
NUM_JOINTS = 20
best_model = models.get(Models.YOLO_NAS_POSE_S, num_classes=NUM_JOINTS, checkpoint_path="C:/Project/super-gradients/checkpoints/training-animalpose-yolo-nas-pose-3lc-2/RUN_20250918_093726_525389/ckpt_best.pth")
predictor = tlc.Predictor(best_model, call_fn="predict", disable_preprocess=True)

def metrics_collection_collate_fn(batch):
    return [sample["image"] for sample in batch]

if __name__ == "__main__":
    batch_size = 32
    num_workers = 4

    start = time.time()
    tlc.collect_metrics(
        table, 
        SuperGradientsPoseMetricsCollector(table),
        predictor,
        split="val",
        constants={"epoch": 0},
        dataloader_args={"batch_size": batch_size, "collate_fn": metrics_collection_collate_fn, "num_workers": num_workers, "persistent_workers": num_workers > 0},
        # collect_aggregates=False,
    )
    end = time.time()
    print(f"Collected metrics for val set (epoch 0) in {end - start} seconds (batch size {batch_size}, workers {num_workers})")

    start = time.time()
    tlc.collect_metrics(
        table, 
        SuperGradientsPoseMetricsCollector(table),
        predictor,
        split="val",
        constants={"epoch": 1},
        dataloader_args={"batch_size": batch_size, "collate_fn": metrics_collection_collate_fn, "num_workers": num_workers, "persistent_workers": num_workers > 0},
        collect_aggregates=False,
    )
    end = time.time()
    print(f"Collected metrics for val set (epoch 1) in {end - start} seconds (batch size {batch_size}, workers {num_workers})")


# Val: 1382
# Train: 3226

# Unfair comparisons, but still interesting. Worker spawn time is dominating, but would not be the case for larger datasets or subsequent calls (due to persistent workers).
# Collected metrics for val set in 98.3032398223877 seconds (batch size 4, workers 0)
# Collected metrics for val set in 82.96371078491211 seconds (batch size 8, workers 0)
# Collected metrics for val set in 44.35213351249695 seconds (batch size 16, workers 0)
# Collected metrics for val set in 40.906614780426025 seconds (batch size 32, workers 0)
# Collected metrics for val set in 37.041160106658936 seconds (batch size 36, workers 0)
# Collected metrics for val set in 128.1499412059784 seconds (batch size 4, workers 4)
# Collected metrics for val set in 89.74156379699707 seconds (batch size 8, workers 4)
# Collected metrics for val set in 119.00389504432678 seconds (batch size 8, workers 8)
# Collected metrics for val set in 100.95760107040405 seconds (batch size 32, workers 8)