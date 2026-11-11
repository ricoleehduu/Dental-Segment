#!/usr/bin/env bash
# Run inference with the released model.
#
#   ./predict.sh /path/to/images /path/to/output
#
# Prerequisites:
#   1. nnUNetv2 installed (validated with 2.6.2; 2.8.x also works)
#   2. this model copied to  $nnUNet_results/Dataset002_ToothFDI/nnUNetTrainer_noMirror__nnUNetPlans__2d
#   3. `python ../code/install_trainer.py` executed once
#   4. nnUNet_raw / nnUNet_preprocessed / nnUNet_results exported
set -euo pipefail

IMAGES="${1:?usage: predict.sh IMAGES_DIR OUTPUT_DIR}"
OUTPUT="${2:?usage: predict.sh IMAGES_DIR OUTPUT_DIR}"

: "${nnUNet_results:?please export nnUNet_results}"

# --disable_tta is REQUIRED: the released checkpoint stores
# inference_allowed_mirroring_axes = (), and vanilla nnU-Net raises
# "ValueError: max() arg is an empty sequence" without this flag.
nnUNetv2_predict \
    -i "$IMAGES" \
    -o "$OUTPUT" \
    -d 2 \
    -c 2d \
    -tr nnUNetTrainer_noMirror \
    -f all \
    -chk checkpoint_best.pth \
    --disable_tta

echo "done -> $OUTPUT"
