from super_gradients.training.utils.callbacks import PhaseCallback, PhaseContext, Phase

class TLCCallback(PhaseCallback):
    def __init__(
        self,
    ):
        super().__init__(phase=Phase.TRAIN_BATCH_END)

    def __call__(self, context: PhaseContext):
        if not context.ddp_silent_mode:
            batch_imgs = context.inputs.cpu().detach().numpy()
            tag = "batch_" + str(context.batch_idx) + "_images"
            context.sg_logger.add_images(tag=tag, images=batch_imgs[: 3], global_step=context.epoch)