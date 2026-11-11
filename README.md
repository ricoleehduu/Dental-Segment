# Dataset002_ToothFDI — FDI tooth segmentation model (artefact)

A single 2D nnU-Net model that segments **52 FDI tooth classes plus background**
(permanent *and* deciduous dentition) on **panoramic radiographs**, trained with
**mirroring disabled**. It is the segmentation model behind the mask-based
validation analyses of the study.

Everything needed to **run** the model is in this folder; the annotated images
themselves are not included (patient data, available under a data-use agreement).

```
model/Dataset002_ToothFDI/nnUNetTrainer_noMirror__nnUNetPlans__2d/
    plans.json, dataset.json, dataset_fingerprint.json
    fold_all/checkpoint_best.pth      <- trained weights (374 MB, sha256 8fa55eee…)
    fold_all/debug.json, progress.png, training_log_*.txt
code/nnUNetTrainer_noMirror.py        <- the trainer that disables mirroring
code/install_trainer.py               <- installs it into the active nnU-Net
scripts/predict.sh, scripts/train.sh
PROVENANCE.md                         <- full training configuration / model card
SHA256SUMS.txt
```

## Requirements

* Python ≥ 3.10, PyTorch
* `pip install nnunetv2` (validated with **2.6.2**; 2.8.x works as well)
* `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` exported

## Quick start (validated end-to-end)

```bash
# 1. install the trainer class that disables mirroring
python code/install_trainer.py

# 2. put the model where nnU-Net expects it
mkdir -p "$nnUNet_results/Dataset002_ToothFDI"
cp -r model/Dataset002_ToothFDI/nnUNetTrainer_noMirror__nnUNetPlans__2d "$nnUNet_results/Dataset002_ToothFDI/"

# 3. predict
nnUNetv2_predict -i /path/to/images -o /path/to/output \
    -d 2 -c 2d -tr nnUNetTrainer_noMirror -f all \
    -chk checkpoint_best.pth --disable_tta
```

`--disable_tta` is **required**. The released checkpoint stores
`inference_allowed_mirroring_axes = ()`, and a stock nnU-Net build raises
`ValueError: max() arg is an empty sequence` when test-time augmentation is
requested with this value. With the flag, prediction of one 482×1024
panoramic image takes ~3–4 s on CPU.

### Alternative without installing any code

Rename the model folder to `nnUNetTrainer__nnUNetPlans__2d` and use
`-tr nnUNetTrainer` (still with `--disable_tta`). The trainer class is then
irrelevant, because the mirroring setting is stored inside the checkpoint.

## Why mirroring is disabled

The labels follow the FDI notation, in which left- and right-side teeth carry
different class codes (e.g. 11 vs 21, 36 vs 46). A horizontal flip would turn a
correctly labelled patch into an image/label pair with inconsistent class
semantics, so mirroring is switched off in the training augmentation and,
consistently, in the test-time augmentation. All other nnU-Net augmentations,
the preprocessing, the loss and the post-processing are the nnU-Net defaults.

## How the model was trained

See `PROVENANCE.md`. Short version: nnU-Net `2d` configuration, patch
512×1024, batch 4, one model (fold `all`), 1000 epochs planned, training
stopped at epoch 768 (best checkpoint at epoch 765), NVIDIA RTX 4060 Ti,
torch 2.12.0+cu126.

The annotated cohort was split 95/5 into training and validation. Five-fold
cross-validation was not used, and the masks come from this single checkpoint
rather than an ensemble. The checkpoint was fit only on the 95 training
images. The five validation images were not used to train it and were not used
to select `checkpoint_best.pth`.

## Redactions

The host name, user name and absolute paths that nnU-Net writes into
`fold_all/debug.json` (and that are quoted in `PROVENANCE.md`) have been replaced
by `<host>`, `<user>` and `<base_dir>` placeholders. No numerical value,
configuration entry, label definition or timing was altered. The unredacted
`debug.json` is retained by the authors and can be provided to the editors on
request.

## Important caveat

Under fold `all`, nnU-Net uses the same 95 training images for its internal
validation loader. The pseudo-Dice values written to `training_log_*.txt`
(≈0.997 at the end) are therefore computed on the training images and must
**not** be reported as segmentation performance. They are not scores on the
five-image validation split.

## Licensing / attribution

* nnU-Net: Apache-2.0, https://github.com/MIC-DKFZ/nnUNet (cite Isensee et al.,
  *Nature Methods* 2021).
* `nnUNetTrainer_noMirror.py`: released with this artefact under Apache-2.0.
* Model weights: released for research use under the terms of the institutional
  data-use agreement of the annotated dataset; please cite the associated paper.
