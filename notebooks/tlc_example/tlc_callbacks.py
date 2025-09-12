from super_gradients.training.utils.callbacks import PhaseContext, Callback
from super_gradients.common.environment.ddp_utils import multi_process_safe
import tlc
class TLCLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
    
    @multi_process_safe
    def on_training_start(self, context: PhaseContext) -> None:
        """
        Called once before start of the first epoch
        At this point, the context argument will have the following attributes:
            - optimizer
            - criterion
            - device
            - experiment_name
            - ckpt_dir
            - net
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics

        The corresponding Phase enum value for this event is Phase.PRE_TRAINING.
        :param context:
        """
        if run := tlc.active_run():
            self.run = run
        else:
            self.run = tlc.init(project_name="animalpose")
        
        self.run.set_parameters(
            {
                "experiment_name": context.experiment_name,
                "ckpt_dir": context.ckpt_dir
            }
        )
        # TODO: log training parameters, etc.

    def on_train_loader_start(self, context: PhaseContext) -> None:
        """
        Called each epoch at the start of train data loader (before getting the first batch).
        At this point, the context argument will have the following attributes:
            - optimizer
            - criterion
            - device
            - experiment_name
            - ckpt_dir
            - net
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
        The corresponding Phase enum value for this event is Phase.TRAIN_EPOCH_START.
        :param context:
        """
        pass

    def on_train_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from data loader and moving it to target device.
        This event triggered AFTER Trainer.pre_prediction_callback call (If it was defined).

        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics

        :param context:
        """
        pass

    def on_train_batch_loss_end(self, context: PhaseContext) -> None:
        """
        Called after model forward and loss computation has been done.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names
        The corresponding Phase enum value for this event is Phase.TRAIN_BATCH_END.

        :param context:
        """
        pass

    def on_train_batch_backward_end(self, context: PhaseContext) -> None:
        """
        Called after loss.backward() method was called for a given batch
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_train_batch_gradient_step_start(self, context: PhaseContext) -> None:
        """
        Called before the graadient step is about to happen.
        Good place to clip gradients (with respect to scaler), log gradients to data ratio, etc.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
        """
        Called after gradient step has been performed. Good place to update LR (for step-based schedulers)
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - inputs
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - criterion
            - device
            - stop_training
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.TRAIN_BATCH_STEP.
        :param context:
        """
        pass

    def on_train_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward/backward/optimizer steps have been performed for a given batch and there is nothing left to do.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_train_loader_end(self, context: PhaseContext) -> None:
        """
        Called each epoch at the end of train data loader (after processing the last batch).
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.TRAIN_EPOCH_END.
        :param context:
        """
        self.run.add_output_value(context.metrics_dict)

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        """
        Called each epoch at the start of validation data loader (before getting the first batch).
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_validation_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from validation loader and moving it to target device.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - inputs
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - criterion
            - device
            - stop_training
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward step / loss / metric computation have been performed for a given batch and there is nothing left to do.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - inputs
            - preds
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.VALIDATION_BATCH_END.
        :param context:
        """
        pass

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        """
        Called each epoch at the end of validation data loader (after processing the last batch).
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.VALIDATION_EPOCH_END.
        :param context:
        """
        self.run.add_output_value(context.metrics_dict)

    def on_validation_end_best_epoch(self, context: PhaseContext) -> None:
        """
        Called each epoch after validation has been performed and the best metric has been achieved.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.VALIDATION_END_BEST_EPOCH.
        :param context:
        """
        pass

    def on_test_loader_start(self, context: PhaseContext) -> None:
        """
        Called once at the start of test data loader (before getting the first batch).
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_test_batch_start(self, context: PhaseContext) -> None:
        """
        Called at each batch after getting batch of data from test loader and moving it to target device.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        :param context:
        """
        pass

    def on_test_batch_end(self, context: PhaseContext) -> None:
        """
        Called after all forward step have been performed for a given batch and there is nothing left to do.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.TEST_BATCH_END.
        :param context:
        """
        pass

    def on_test_loader_end(self, context: PhaseContext) -> None:
        """
        Called once at the end of test data loader (after processing the last batch).
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.TEST_END.
        :param context:
        """
        pass

    def on_average_best_models_validation_start(self, context: PhaseContext) -> None:
        """
        Called once after the test was end before the training loop has finished.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.AVERAGE_BEST_MODELS_VALIDATION_START.
        :param context:
        """
        pass

    def on_average_best_models_validation_end(self, context: PhaseContext) -> None:
        """
        Called once after the average model validation has finished.
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_dict
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.AVERAGE_BEST_MODELS_VALIDATION_START.
        :param context:
        """
        pass

    def on_training_end(self, context: PhaseContext) -> None:
        """
        Called once after the training loop has finished (Due to reaching optimization criterion or because of an error.)
        At this point, the context argument will have the following attributes:
            - epoch
            - batch_idx
            - optimizer
            - inputs
            - preds
            - target
            - metrics_compute_fn
            - loss_avg_meter
            - loss_log_items
            - criterion
            - device
            - stop_training
            - experiment_name
            - ckpt_dir
            - net
            - lr_warmup_epochs
            - sg_logger
            - train_loader
            - valid_loader
            - training_params
            - ddp_silent_mode
            - checkpoint_params
            - arch_params
            - metric_to_watch
            - valid_metrics
            - loss_logging_items_names

        The corresponding Phase enum value for this event is Phase.POST_TRAINING.
        :param context:
        """
        self.run.set_status_completed()