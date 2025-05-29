# from pytorch_lightning.utilities import rank_zero_only
# import torch
# from dataloader.FireSpreadDataModule import FireSpreadDataModule
# from pytorch_lightning.cli import LightningCLI
# from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
# from models import BaseModel
# import wandb
# import os

# from dataloader.FireSpreadDataset import FireSpreadDataset
# from dataloader.utils import get_means_stds_missing_values

# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# torch.set_float32_matmul_precision('high')


# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("trainer.default_root_dir",
#                               "trainer.logger.init_args.save_dir")
#         parser.link_arguments("model.class_path",
#                               "trainer.logger.init_args.name")
#         parser.add_argument("--do_train", type=bool,
#                             help="If True: skip training the model.")
#         parser.add_argument("--do_predict", type=bool,
#                             help="If True: compute predictions.")
#         parser.add_argument("--do_test", type=bool,
#                             help="If True: compute test metrics.")
#         parser.add_argument("--do_validate", type=bool,
#                             default=False, help="If True: compute val metrics.")
#         parser.add_argument("--ckpt_path", type=str, default=None,
#                             help="Path to checkpoint to load for resuming training, for testing and predicting.")

#     def before_instantiate_classes(self):
#         # The number of features is only known inside the data module, but we need that info to instantiate the model.
#         # Since datamodule and model are instantiated at the same time with LightningCLI, we need to set the number of features here.
#         n_features = FireSpreadDataset.get_n_features(
#             self.config.data.n_leading_observations,
#             self.config.data.features_to_keep,
#             self.config.data.remove_duplicate_features)
#         self.config.model.init_args.n_channels = n_features

#         # The exact positive class weight changes with the data fold in the data module, but the weight is needed to instantiate the model.
#         # Non-fire pixels are marked as missing values in the active fire feature, so we simply use that to compute the positive class weight.
#         train_years, _, _ = FireSpreadDataModule.split_fires(
#             self.config.data.data_fold_id)
#         _, _, missing_values_rates = get_means_stds_missing_values(train_years)
#         fire_rate = 1 - missing_values_rates[-1]
#         pos_class_weight = float(1 / fire_rate)

#         self.config.model.init_args.pos_class_weight = pos_class_weight

#     def before_fit(self):
#         self.wandb_setup()

#     def before_test(self):
#         self.wandb_setup()

#     def before_validate(self):
#         self.wandb_setup()

#     @rank_zero_only
#     def wandb_setup(self):
#         """
#         Save the config used by LightningCLI to disk, then save that file to wandb.
#         Using wandb.config adds some strange formating that means we'd have to do some 
#         processing to be able to use it again as CLI input.

#         Also define min and max metrics in wandb, because otherwise it just reports the 
#         last known values, which is not what we want.
#         """
#         config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

#         cfg_string = self.parser.dump(self.config, skip_none=False)
#         with open(config_file_name, "w") as f:
#             f.write(cfg_string)
#         wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
#         wandb.define_metric("train_loss_epoch", summary="min")
#         wandb.define_metric("val_loss", summary="min")
#         wandb.define_metric("train_f1_epoch", summary="max")
#         wandb.define_metric("val_f1", summary="max")


# def main():
#     cli = MyLightningCLI(
#         BaseModel,
#         FireSpreadDataModule,
#         subclass_mode_model=True,
#         save_config_kwargs={"overwrite": True},
#         parser_kwargs={"parser_mode": "yaml"},
#         run=False
#     )

#     cli.wandb_setup()

#     # Train (if requested)
#     if cli.config.do_train:
#         cli.trainer.fit(cli.model, cli.datamodule)

#     # Set ckpt to last.ckpt if it exists and logger is available
#     ckpt = cli.config.ckpt_path
#     log_dir = getattr(cli.trainer.logger, "log_dir", None)
#     if ckpt is None:
#         base_dir = log_dir or cli.trainer.default_root_dir
#         #ckpt_path = os.path.join(base_dir, "checkpoints", "last.ckpt")
#         ckpt_path = os.path.join("checkpoints", "last.ckpt")
#         if os.path.exists(ckpt_path):
#             print(f"[INFO] Using last checkpoint from {ckpt_path}")
#             ckpt = ckpt_path
#         else:
#             print(f"[WARN] No last checkpoint found at {ckpt_path}, using current model state.")
#             ckpt = None

#     # Clear stale checkpoint paths
#     cli.config.ckpt_path = None
#     cli.config.trainer.ckpt_path = None

#     # Validate
#     if cli.config.do_validate:
#         cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

#     # Test
#     if cli.config.do_test:
#         cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

#     # Predict
#     if cli.config.do_predict:
#         prediction_output = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)
#         x_af = torch.cat([tup[0][:, -1, :, :].squeeze() for tup in prediction_output], dim=0)
#         y = torch.cat([tup[1] for tup in prediction_output], dim=0)
#         y_hat = torch.cat([tup[2] for tup in prediction_output], dim=0)

#         fire_masks_combined = torch.cat(
#             [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], dim=0
#         )

#         predictions_file_name = os.path.join(
#             cli.config.trainer.default_root_dir,
#             f"predictions_{wandb.run.id}.pt"
#         )
#         torch.save(fire_masks_combined, predictions_file_name)
#         print(f"[INFO] Saved predictions to: {predictions_file_name}")




# if __name__ == "__main__":
#     main()

from pytorch_lightning.utilities import rank_zero_only
import torch
from dataloader.FireSpreadDataModule import FireSpreadDataModule
from pytorch_lightning.cli import LightningCLI
from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
from models import BaseModel
import wandb
import os
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                            "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path",
                            "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool,
                          help="If True: skip training the model.")
        parser.add_argument("--do_predict", type=bool,
                          help="If True: compute predictions.")
        parser.add_argument("--do_test", type=bool,
                          help="If True: compute test metrics.")
        parser.add_argument("--do_validate", type=bool,
                          default=False, help="If True: compute val metrics.")
        parser.add_argument("--ckpt_path", type=str, default=None,
                          help="Path to checkpoint to load for resuming training, for testing and predicting.")

    def before_instantiate_classes(self):
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features)
        self.config.model.init_args.n_channels = n_features

        train_years, _, _ = FireSpreadDataModule.split_fires(
            self.config.data.data_fold_id)
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)

        self.config.model.init_args.pos_class_weight = pos_class_weight

    def after_instantiate_classes(self):
        """Called after datamodule and model are instantiated"""
        # Force setup of the datamodule
        self.datamodule.setup('fit')
        
        # Print dataset statistics immediately after setup
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(f"Train samples: {len(self.datamodule.train_dataset)}")
        print(f"Val samples: {len(self.datamodule.val_dataset)}")
        print(f"Test samples: {len(self.datamodule.test_dataset)}")
        print("="*80 + "\n")

    def before_fit(self):
        self.wandb_setup()
        self.print_dataset_stats()
        
        print("\n=== BEFORE FIT ===")
        if hasattr(self.datamodule, 'train_dataset'):
            print(f"Train samples: {len(self.datamodule.train_dataset)}")
        else:
            print("Train dataset not available yet!")

    def before_test(self):
        self.wandb_setup()
        self.print_dataset_stats()

    def before_validate(self):
        self.wandb_setup()
        self.print_dataset_stats()

    @rank_zero_only
    def print_dataset_stats(self):
        """Print dataset statistics in a clear format"""
        if hasattr(self, 'datamodule') and self.datamodule is not None:
            rank_zero_info("\n" + "="*80)
            rank_zero_info("DATASET STATISTICS")
            rank_zero_info("="*80)
            if hasattr(self.datamodule, 'train_dataset') and self.datamodule.train_dataset is not None:
                rank_zero_info(f"Train samples: {len(self.datamodule.train_dataset)}")
            if hasattr(self.datamodule, 'val_dataset') and self.datamodule.val_dataset is not None:
                rank_zero_info(f"Val samples: {len(self.datamodule.val_dataset)}")
            if hasattr(self.datamodule, 'test_dataset') and self.datamodule.test_dataset is not None:
                rank_zero_info(f"Test samples: {len(self.datamodule.test_dataset)}")
            rank_zero_info("="*80 + "\n")

    @rank_zero_only
    def wandb_setup(self):
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
        cfg_string = self.parser.dump(self.config, skip_none=False)
        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")


def main():
    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False
    )
    
    # Force setup of the datamodule
    cli.datamodule.setup('fit')
    
    # Get statistics directly from dataset objects
    def get_stats(dataset):
        if dataset is None:
            return (0, 0)
        # Count fire events by summing the number of fires per year
        fire_count = sum(len(fires) for fires in dataset.imgs_per_fire.values())
        return (fire_count, len(dataset))
    
    train_fires, train_samples = get_stats(cli.datamodule.train_dataset)
    val_fires, val_samples = get_stats(cli.datamodule.val_dataset)
    test_fires, test_samples = get_stats(cli.datamodule.test_dataset)
    
    print("\n=== ACTUAL DATASET STATISTICS ===")
    print(f"Train fires: {train_fires} - Samples: {train_samples}")
    print(f"Val fires: {val_fires} - Samples: {val_samples}")
    print(f"Test fires: {test_fires} - Samples: {test_samples}")
    print("===============================\n")
            
    cli.wandb_setup()

    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule)

    ckpt = cli.config.ckpt_path
    log_dir = getattr(cli.trainer.logger, "log_dir", None)
    if ckpt is None:
        base_dir = log_dir or cli.trainer.default_root_dir
        ckpt_path = os.path.join("checkpoints", "last.ckpt")
        if os.path.exists(ckpt_path):
            print(f"[INFO] Using last checkpoint from {ckpt_path}")
            ckpt = ckpt_path
        else:
            print(f"[WARN] No last checkpoint found at {ckpt_path}, using current model state.")
            ckpt = None

    cli.config.ckpt_path = None
    cli.config.trainer.ckpt_path = None

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:
        prediction_output = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat([tup[0][:, -1, :, :].squeeze() for tup in prediction_output], dim=0)
        y = torch.cat([tup[1] for tup in prediction_output], dim=0)
        y_hat = torch.cat([tup[2] for tup in prediction_output], dim=0)

        fire_masks_combined = torch.cat(
            [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], dim=0
        )

        predictions_file_name = os.path.join(
            cli.config.trainer.default_root_dir,
            f"predictions_{wandb.run.id}.pt"
        )
        torch.save(fire_masks_combined, predictions_file_name)
        print(f"[INFO] Saved predictions to: {predictions_file_name}")

if __name__ == "__main__":
    main()