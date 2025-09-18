import os
import sys
stdout = sys.stdout
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN
from super_gradients.training.transforms.keypoints import (
    KeypointsRandomHorizontalFlip,
    KeypointsHSV,
    KeypointsBrightnessContrast,
    KeypointsMosaic,
    KeypointsRandomAffineTransform,
    KeypointsLongestMaxSize,
    KeypointsPadIfNeeded,
    KeypointsImageStandardize,
    KeypointsRemoveSmallObjects,
)
from torch.utils.data import DataLoader
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.metrics import PoseEstimationMetrics
from tlc_callbacks import TLCLoggingCallback
from tlc_pose_dataset import TLCPoseEstimationDataset
from tlc.core import Table
import tlc
from pose_metrics_collector import SuperGradientsPoseMetricsCollector

sys.stdout = stdout

# Constants
CHECKPOINT_DIR = "checkpoints"
IMAGE_SIZE = 640
OKS_SIGMAS = [0.07] * 20
FLIP_INDEXES = [1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19]
KEYPOINT_NAMES = [
    "left eye",
    "right eye",
    "nose",
    "left earbase",
    "right earbase",
    "L_F_elbow",
    "R_F_elbow",
    "L_B_elbow",
    "R_B_elbow",
    "L_F_knee",
    "R_F_knee",
    "L_B_knee",
    "R_B_knee",
    "L_F_paw",
    "R_F_paw",
    "L_B_paw",
    "R_B_paw",
    "throat",
    "withers",
    "tailbase",
]
NUM_JOINTS = len(KEYPOINT_NAMES)
KEYPOINT_COLORS = [
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
]
EDGE_LINKS = [
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 3],
    [1, 4],
    [2, 17],
    [18, 19],
    [5, 9],
    [6, 10],
    [7, 11],
    [8, 12],
    [9, 13],
    [10, 14],
    [11, 15],
    [12, 16],
]
EDGE_COLORS = [
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

def create_transforms(flip_indexes, image_size):
    keypoints_random_horizontal_flip = KeypointsRandomHorizontalFlip(flip_index=flip_indexes, prob=0.5)
    keypoints_hsv = KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20)
    keypoints_brightness_contrast = KeypointsBrightnessContrast(prob=0.5, brightness_range=[0.8, 1.2], contrast_range=[0.8, 1.2])
    # keypoints_mosaic = KeypointsMosaic(prob=0.8)
    keypoints_random_affine_transform = KeypointsRandomAffineTransform(
    max_rotation=0, min_scale=0.5, max_scale=1.5, max_translate=0.1, image_pad_value=127, mask_pad_value=1, prob=0.75, interpolation_mode=[0, 1, 2, 3, 4]
)
    keypoints_longest_max_size = KeypointsLongestMaxSize(max_height=image_size, max_width=image_size)
    keypoints_pad_if_needed = KeypointsPadIfNeeded(
    min_height=image_size, min_width=image_size, image_pad_value=[127, 127, 127], mask_pad_value=1, padding_mode="bottom_right"
)
    keypoints_image_standardize = KeypointsImageStandardize(max_value=255)
    keypoints_remove_small_objects = KeypointsRemoveSmallObjects(min_instance_area=1, min_visible_keypoints=1)

    train_transforms = [
    keypoints_random_horizontal_flip,
    keypoints_hsv,
    keypoints_brightness_contrast,
    # You can enable mosaic augmentation if you want. This usually improves the metric at the cost of increased training time
    # keypoints_mosaic,
    keypoints_random_affine_transform,
    keypoints_longest_max_size,
    keypoints_pad_if_needed,
    keypoints_image_standardize,
    keypoints_remove_small_objects,
]

    val_transforms = [
    keypoints_longest_max_size,
    keypoints_pad_if_needed,
    keypoints_image_standardize,
]
    
    return train_transforms,val_transforms

def metrics_collection_collate_fn(batch):
    return [sample["image"] for sample in batch]

if __name__ == "__main__":
    
    # Load tables  # TODO: create tables
    train_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/train")
    val_table = Table.from_url("C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/animalpose/datasets/animalpose/tables/val")

    # Create datasets
    train_transforms, val_transforms = create_transforms(FLIP_INDEXES, IMAGE_SIZE)
    train_dataset = TLCPoseEstimationDataset(train_table, train_transforms)
    val_dataset = TLCPoseEstimationDataset(val_table, val_transforms)

    # Create dataloaders
    train_dataloader_params = {
        "shuffle": True,
        "batch_size": 24,
        "drop_last": True,
        "pin_memory": False,
        "collate_fn": YoloNASPoseCollateFN(),
        "num_workers": 8,
        "persistent_workers": True,
    }
    val_dataloader_params = {
        "shuffle": False,
        "batch_size": 24,
        "drop_last": True,
        "pin_memory": False,
        "collate_fn": YoloNASPoseCollateFN(),
        "num_workers": 8,
        "persistent_workers": True,
    }

    train_dataloader = DataLoader(train_dataset, **train_dataloader_params)
    val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

    # Instantiate the model
    yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_S, num_classes=NUM_JOINTS, pretrained_weights="coco_pose").cuda()

    # Instantiate the trainer
    trainer = Trainer(
        experiment_name="training-animalpose-yolo-nas-pose-3lc-2",
        ckpt_root_dir=CHECKPOINT_DIR,
    )

    # Create callbacks
    post_prediction_callback = YoloNASPosePostPredictionCallback(
        pose_confidence_threshold=0.01,
        nms_iou_threshold=0.7,
        pre_nms_max_predictions=300,
        post_nms_max_predictions=30,
    )

    metrics = PoseEstimationMetrics(
        num_joints=NUM_JOINTS,
        oks_sigmas=OKS_SIGMAS,
        max_objects_per_image=30,
        post_prediction_callback=post_prediction_callback,
    )

    visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
        keypoint_colors=KEYPOINT_COLORS,
        edge_colors=EDGE_COLORS,
        edge_links=EDGE_LINKS,
        loss_to_monitor="YoloNASPoseLoss/loss",
        max=True,
        freq=1,
        max_images=16,
        enable_on_train_loader=True,
        enable_on_valid_loader=True,
        post_prediction_callback=post_prediction_callback,
    )

    early_stop = EarlyStop(
        phase=Phase.VALIDATION_EPOCH_END,
        monitor="AP",
        mode="max",
        min_delta=0.0001,
        patience=100,
        verbose=True,
    )

    tlc_logging_callback = TLCLoggingCallback(project_name="animalpose")

    # Create training params
    train_params = {
        "warmup_mode": "LinearBatchLRWarmup",
        "warmup_initial_lr": 1e-8,
        "lr_warmup_epochs": 2,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.05,
        "max_epochs": 5,
        "zero_weight_decay_on_bias_and_bn": True,
        "batch_accumulate": 1,
        "average_best_models": True,
        "save_ckpt_epoch_list": [],
        "loss": "yolo_nas_pose_loss",
        "criterion_params": {
            "oks_sigmas": OKS_SIGMAS,
            "classification_loss_weight": 1.0,
            "classification_loss_type": "focal",
            "regression_iou_loss_type": "ciou",
            "iou_loss_weight": 2.5,
            "dfl_loss_weight": 0.01,
            "pose_cls_loss_weight": 1.0,
            "pose_reg_loss_weight": 34.0,
            "pose_classification_loss_type": "focal",
            "rescale_pose_loss_with_assigned_score": True,
            "assigner_multiply_by_pose_oks": True,
        },
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.000001},
        "ema": True,
        "ema_params": {"decay": 0.997, "decay_type": "threshold"},
        "mixed_precision": True,
        "sync_bn": False,
        "valid_metrics_list": [metrics],
        "phase_callbacks": [visualization_callback, early_stop, tlc_logging_callback],
        "pre_prediction_callback": None,
        "metric_to_watch": "AP",
        "greater_metric_to_watch_is_better": True,
    }

    # Train the model
    trainer.train(
        model=yolo_nas_pose,
        training_params=train_params,
        train_loader=train_dataloader,
        valid_loader=val_dataloader
    )

    # Collect per-sample metrics on validation set using best model
    best_model = models.get(Models.YOLO_NAS_POSE_S, num_classes=NUM_JOINTS, checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))
    predictor = tlc.Predictor(best_model, call_fn="predict", disable_preprocess=True)

    tlc.collect_metrics(
        val_table, 
        SuperGradientsPoseMetricsCollector(val_table),
        predictor,
        split="val",
        constants={"epoch": trainer.max_epochs},
        dataloader_args={"batch_size": 32, "collate_fn": metrics_collection_collate_fn, "num_workers": 8, "persistent_workers": True},
        collect_aggregates=False,
    )

    tlc.collect_metrics(
        train_table, 
        SuperGradientsPoseMetricsCollector(val_table),
        predictor,
        split="train",
        constants={"epoch": trainer.max_epochs},
        dataloader_args={"batch_size": 32, "collate_fn": metrics_collection_collate_fn, "num_workers": 8, "persistent_workers": True},
        collect_aggregates=False,
    )
