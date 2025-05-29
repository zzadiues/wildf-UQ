#!/bin/bash
set -e  # Stop on first error

FOLD_ID=10
SEEDS=(42)

declare -A FEATURE_SETS
#FEATURE_SETS["vegetation"]='[0, 1, 2, 3, 4, 38, 39]'
FEATURE_SETS["weather"]='[5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 39]'
FEATURE_SETS["topography"]='[12, 13, 14, 22, 39]'
FEATURE_SETS["landcover"]='[16, 22, 39]'

for SEED in "${SEEDS[@]}"; do
  for FEATURE_TYPE in "${!FEATURE_SETS[@]}"; do
    echo "🚀 Training model on Fold ${FOLD_ID} with using ${FEATURE_TYPE} features"

    CHECKPOINT_DIR="checkpoints_variants/fold${FOLD_ID}_seed${SEED}_${FEATURE_TYPE}"
    mkdir -p "$CHECKPOINT_DIR"

    TRAINER_CONFIG_TMP="cfgs/tmp_trainer_gpu_fold${FOLD_ID}_seed${SEED}_${FEATURE_TYPE}.yaml"
    cp cfgs/trainer_single_gpu.yaml "$TRAINER_CONFIG_TMP"

    echo "
callbacks:
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      save_last: true
      every_n_train_steps: 5000
      save_top_k: -1
      dirpath: ${CHECKPOINT_DIR}
      filename: step{step}_fold${FOLD_ID}_seed${SEED}_${FEATURE_TYPE}
" >> "$TRAINER_CONFIG_TMP"

    WANDB_MODE=disabled python3 train.py \
      --seed_everything=${SEED} \
      -c cfgs/UTAE/all_features.yaml \
      --trainer "$TRAINER_CONFIG_TMP" \
      --data cfgs/data_monotemporal_full_features.yaml \
      --data.data_dir=/mnt/c/Users/chakr/OneDrive/Desktop/wildf \
      --data.data_fold_id=${FOLD_ID} \
      --data.features_to_keep="${FEATURE_SETS[$FEATURE_TYPE]}" \
      --data.n_leading_observations=5 \
      --data.batch_size=16 \
      --data.return_doy=True \
      --trainer.max_steps=10000 \
      --do_train=True \
      --do_test=True \
      --do_predict=True

    echo "✅ Finished training for Fold ${FOLD_ID} using ${FEATURE_TYPE}"
    echo ""
  done
done


'''
Purpose	features_to_keep
Vegetation	[0, 1, 2, 3, 4, 38, 39]
Weather only	[5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22, 39]
Topography only	[12, 13, 14, 22, 39]
Landcover only	[16, 22, 39]
'''