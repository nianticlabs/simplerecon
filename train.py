""" 
    Trains a DepthModel model. Uses an MVS dataset from datasets.

    - Outputs logs and checkpoints to opts.log_dir/opts.name
    - Supports mixed precision training by setting '--precision 16'

    We train with a batch_size of 16 with 16-bit precision on two A100s.

    Example command to train with two GPUs
        python train.py --name HERO_MODEL \
                    --log_dir logs \
                    --config_file configs/models/hero_model.yaml \
                    --data_config configs/data/scannet_default_train.yaml \
                    --gpus 2 \
                    --batch_size 16;
                    
"""


import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader

import options
from experiment_modules.depth_model import DepthModel
from utils.generic_utils import copy_code_state
from utils.dataset_utils import get_dataset


def main(opts):

    # set seed
    pl.seed_everything(opts.random_seed) 
    

    if opts.load_weights_from_checkpoint is not None:
        model = DepthModel.load_from_checkpoint(
            opts.load_weights_from_checkpoint,
            opts=opts,
            args=None
        )
    else:
        # load model using read options
        model = DepthModel(opts)

    # load dataset and dataloaders
    dataset_class, _ = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    train_dataset = dataset_class(
                        opts.dataset_path,
                        split="train",
                        mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                        num_images_in_tuple=opts.num_images_in_tuple,
                        tuple_info_file_location=opts.tuple_info_file_location,
                        image_width=opts.image_width,
                        image_height=opts.image_height,
                        shuffle_tuple=opts.shuffle_tuple,
                    )

    train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=opts.batch_size,
                                shuffle=True,
                                num_workers=opts.num_workers,
                                pin_memory=True,
                                drop_last=True,
                                persistent_workers=True,
                            )

    val_dataset = dataset_class(
                        opts.dataset_path,
                        split="val",
                        mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                        num_images_in_tuple=opts.num_images_in_tuple,
                        tuple_info_file_location=opts.tuple_info_file_location,
                        image_width=opts.image_width,
                        image_height=opts.image_height,
                        include_full_res_depth=opts.high_res_validation,
                    )

    val_dataloader = DataLoader(
                            val_dataset,
                            batch_size=opts.val_batch_size,
                            shuffle=False,
                            num_workers=opts.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            persistent_workers=True,
                        )

    # set up a tensorboard logger through lightning
    logger = TensorBoardLogger(save_dir=opts.log_dir, name=opts.name)

    # This will copy a snapshot of the code (minus whatever is in .gitignore) 
    # into a folder inside the main log directory.
    copy_code_state(path=os.path.join(logger.log_dir, "code"))

    # dumping a copy of the config to the directory for easy(ier) 
    # reproducibility.
    options.OptionsHandler.save_options_as_yaml(
                                    os.path.join(logger.log_dir, "config.yaml"),
                                    opts,
                            )

    # set a checkpoint callback for lignting to save model checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                save_last=True,
                                save_top_k=1,
                                verbose=True,
                                monitor='val/loss',
                                mode='min',
                            )

    
    # keep track of changes in learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # allowing the lightning DDPPlugin to ignore unused params.
    find_unused_parameters = (opts.matching_encoder_type == "unet_encoder")
    
    trainer = pl.Trainer(
                        gpus=opts.gpus,
                        log_every_n_steps=opts.log_interval,
                        val_check_interval=opts.val_interval,
                        limit_val_batches=opts.val_batches,
                        max_steps=opts.max_steps,
                        precision=opts.precision,
                        benchmark=True,
                        logger=logger,
                        sync_batchnorm=False,
                        callbacks=[checkpoint_callback, lr_monitor],
                        num_sanity_val_steps=opts.num_sanity_val_steps,
                        strategy=DDPPlugin(
                            find_unused_parameters=find_unused_parameters
                        ),
                        resume_from_checkpoint=opts.resume,
                    )

    # start training
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
