#!/usr/bin/env bash
# Re-train the model from scratch (requires the annotated dataset in nnU-Net raw format).
#
#   ./train.sh
#
# Prerequisites:
#   1. dataset in  $nnUNet_raw/Dataset002_ToothFDI/{imagesTr,labelsTr,dataset.json}
#   2. nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
#   3. `python ../code/install_trainer.py` executed once
set -euo pipefail

: "${nnUNet_results:?please export nnUNet_results}"

# single model trained on all annotated cases (fold "all"), identical to the released checkpoint
nnUNetv2_train 2 2d all -tr nnUNetTrainer_noMirror
