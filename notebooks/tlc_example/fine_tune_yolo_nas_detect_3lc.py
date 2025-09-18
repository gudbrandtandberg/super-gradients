import os
import tlc

from tlc_detection_dataset import TLCDetectionDataset
from tlc_detection_metrics_collector import SuperGradientsDetectionMetricsCollector

from torch.utils.data import DataLoader
from super_gradients.training.utils.collate_fn.detection_collate_fn import DetectionCollateFN
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.common.object_names import Models

if __name__ == "__main__":
    train_table = tlc.Table.from_url('/Users/frederik/Library/Application Support/3LC/projects/coco8-YOLO/datasets/coco8-train/tables/initial')
    val_table = tlc.Table.from_url('/Users/frederik/Library/Application Support/3LC/projects/coco8-YOLO/datasets/coco8-val/tables/initial')

    train_table = train_table
    val_table = val_table

    num_classes = len(train_table.get_simple_value_map("bbs.bb_list.label").keys())

    train_dataset = TLCDetectionDataset(train_table)
    val_dataset = TLCDetectionDataset(val_table)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=DetectionCollateFN())
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=DetectionCollateFN())

    model = models.get(Models.YOLO_NAS_S, num_classes=num_classes, pretrained_weights="coco")

    trainer = Trainer(
        experiment_name="fine-tune-yolo-nas-detect-3lc",
        ckpt_root_dir="checkpoints",
    )

    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    train_params = {
        "warmup_initial_lr": 1e-5,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.5,
        "optimizer": "AdamW",
        "zero_weight_decay_on_bias_and_bn": True,
        "lr_warmup_epochs": 1,
        "warmup_mode": "LinearEpochLRWarmup",
        "optimizer_params": {"weight_decay": 0.0001},
        "ema": False,
        "average_best_models": False,
        "ema_params": {"beta": 25, "decay_type": "exp"},
        "max_epochs": 5,
        "mixed_precision": True,
        "loss": PPYoloELoss(use_static_assigner=False, num_classes=num_classes, reg_max=16),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=num_classes,
                normalize_targets=True,
                include_classwise_ap=True,
                class_names=list(train_table.get_simple_value_map("bbs.bb_list.label").keys()),
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
            )
        ],
        "metric_to_watch": "mAP@0.50",
    }

    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
    )

    # best_model = models.get(Models.YOLO_NAS_S, num_classes=num_classes, checkpoint_path=os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth"))
    
    # best_model.set_dataset_processing_params(
    #     class_names=list(train_table.get_simple_value_map("bbs.bb_list.label").keys()),
    #     image_processor=None,
    #     iou=0.5,
    #     conf=0.05,
    # )
    
    predictor = tlc.Predictor(model, call_fn="predict", disable_preprocess=True)

    tlc.collect_metrics(
        val_table.map(lambda row: row["image"]),
        SuperGradientsDetectionMetricsCollector(val_table),
        predictor,
        split="val",
        constants={"epoch": trainer.max_epochs},
        dataloader_args={"batch_size": 4},
        collect_aggregates=False,
    )