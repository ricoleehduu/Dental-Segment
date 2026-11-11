# 训练溯源（PROVENANCE）

本文件记录 `fold_all/checkpoint_best.pth` 的训练现场，数据全部取自随包附带的
`debug.json`、`plans.json`、`dataset.json` 和 `training_log_2026_6_24_20_42_00.txt`。

## 数据

| 项 | 值 |
|---|---|
| 数据集 | `Dataset002_ToothFDI` — 全景片（2D，PNG，单通道 X-ray） |
| 标注 | 53 个类别：背景 0 + 52 个 FDI 牙位（11–48 恒牙 + 51–85 乳牙） |
| 训练例数 | 95（`dataset.json` 的 `numTraining`） |
| 划分 | 研究层面为 95 张训练 / 5 张验证，未做五折。本 checkpoint 为 `fold = all`，只在 95 张训练集上拟合；nnU-Net 日志里的验证与这 5 张无关 |

## 训练配置（plans.json / debug.json）

| 项 | 值 |
|---|---|
| nnU-Net 配置 | `2d`（`nnUNetPlans`，预处理 `DefaultPreprocessor`，ZScore 归一化） |
| patch size | 512 × 1024 |
| batch size | 4（batch_dice = True） |
| 网络 | `PlainConvUNet`，8 个 stage，通道 32→64→128→256→512→512→512→512，InstanceNorm2d + LeakyReLU |
| 深监督 | 开启 |
| 损失 | Dice + CE（`DC_and_CE_loss`） |
| 优化器 | SGD，nesterov，momentum 0.99，weight decay 3e-5，初始 lr 0.01（poly） |
| 迭代 | 250 iterations/epoch，计划 1000 epochs，每 50 epoch 保存 |
| 镜像增强 | **关闭**（checkpoint 中 `inference_allowed_mirroring_axes = ()`） |

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| PyTorch | 2.12.0+cu126，cuDNN 9.10.02 |
| 训练主机 | `<host>`，工作目录 `<base_dir>`（已脱敏，原值见 `debug.json` 说明） |
| 训练时间 | 2026-06-24 20:43 → 2026-06-25 17:39（约 21 小时） |
| 实际停止 | 第 768 epoch（计划 1000，被中断） |
| 最佳 checkpoint | 第 765 epoch（`checkpoint_best.pth`，训练集自评 EMA pseudo-Dice 0.9971） |

> 「训练集自评」的含义：nnU-Net 在 `fold=all` 时把全部图像同时用于训练和验证
> （源码注释原文："if fold==all then we use all images for training and validation"）。
> 因此日志里的 pseudo-Dice 不能作为分割性能指标；本包不提供留出验证结果。

## 校验和

```
checkpoint_best.pth  sha256 8fa55eee8cb2b4c098dec590ca326e2a6c335872a2d9597f04e8dc9dee2c48e4
```

## 复现训练

需要原始标注数据（未随包发布）。命令见 `scripts/train.sh`：

```bash
nnUNetv2_train 2 2d all -tr nnUNetTrainer_noMirror
```

`code/nnUNetTrainer_noMirror.py` 与本包 checkpoint 记录的
`inference_allowed_mirroring_axes = ()` 一致；其他增强、plans、预处理均未改动。
