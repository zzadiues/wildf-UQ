

#!/bin/bash
set -e  # Stop on first error

SEEDS=(42 1337 2024 31415 777)

for FOLD_ID in {0..11}; do
  for SEED in "${SEEDS[@]}"; do
    echo "🚀 Training Deep Ensemble model on Fold ${FOLD_ID} with seed ${SEED}"

    # === Define checkpoint directory for this ensemble member ===
    CHECKPOINT_DIR="checkpoints/ensemble_fold${FOLD_ID}_seed${SEED}"
    mkdir -p "$CHECKPOINT_DIR"

    # === Create temporary trainer YAML ===
    TRAINER_CONFIG_TMP="cfgs/tmp_trainer_gpu_fold${FOLD_ID}_seed${SEED}.yaml"
    cp cfgs/trainer_single_gpu.yaml $TRAINER_CONFIG_TMP

    # === Add checkpoint saving callback ===
    echo "
callbacks:
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      save_last: true
      every_n_train_steps: 5000
      save_top_k: -1
      dirpath: ${CHECKPOINT_DIR}
      filename: step{step}_fold${FOLD_ID}_seed${SEED}
" >> $TRAINER_CONFIG_TMP

    # === Launch training with seed set via CLI ===
    WANDB_MODE=disabled python3 train.py \
      --seed_everything=${SEED} \
      -c cfgs/UTAE/all_features.yaml \
      --trainer $TRAINER_CONFIG_TMP \
      --data cfgs/data_monotemporal_full_features.yaml \
      --data.data_dir=/mnt/c/Users/chakr/OneDrive/Desktop/wildf \
      --data.data_fold_id=${FOLD_ID} \
      --data.features_to_keep="[0, 1, 2, 3, 4, 38, 39]" \
      --data.n_leading_observations=5 \
      --data.batch_size=16 \
      --data.return_doy=True \
      --trainer.max_steps=10000 \
      --do_train=True \
      --do_test=True \
      --do_predict=True

    echo "✅ Finished Ensemble Seed ${SEED} for Fold ${FOLD_ID}"
    echo ""
  done
done
