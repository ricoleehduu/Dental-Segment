"""Install nnUNetTrainer_noMirror into the active nnU-Net installation.

Usage:
    python install_trainer.py
"""
import shutil
from pathlib import Path

import nnunetv2

src = Path(__file__).resolve().with_name("nnUNetTrainer_noMirror.py")
dst_dir = Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer" / "variants" / "data_augmentation"
dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / src.name
shutil.copy2(src, dst)
print(f"installed: {dst}")
print("check with: nnUNetv2_train -h | findstr noMirror")
