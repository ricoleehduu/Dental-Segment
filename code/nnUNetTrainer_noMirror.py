"""Trainer used for the panoramic-radiograph FDI tooth segmentation model.

Mirroring (left/right flip) is disabled in both the training augmentation and the
test-time augmentation. The labels follow the FDI notation, in which left- and
right-side teeth carry different class codes (e.g. 11 vs 21, 36 vs 46), so a
horizontal flip would turn a correctly labelled training patch into an
image/label pair whose class semantics are inconsistent.

This class reproduces the checkpoint shipped with this artefact: the released
``checkpoint_best.pth`` records ``inference_allowed_mirroring_axes = ()``,
i.e. mirroring was switched off for training and for inference.

Install with ``code/install_trainer.py`` (or copy this file into
``nnunetv2/training/nnUNetTrainer/variants/data_augmentation/``), then use
``-tr nnUNetTrainer_noMirror``.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_noMirror(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Empty tuple = no mirroring. `mirror_axes` is the single source of
        # self.inference_allowed_mirroring_axes (set in the parent method), which is
        # stored in the checkpoint and used by the predictor for TTA.
        mirror_axes = ()
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
