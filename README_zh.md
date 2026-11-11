# 交付包说明（中文）

这个包 = **权重 + 代码 + 训练溯源**，用来回应审稿人「weights / configuration / inference settings 没提供」。

## 里面有什么

```
model/Dataset002_ToothFDI/nnUNetTrainer_noMirror__nnUNetPlans__2d/
    plans.json / dataset.json / dataset_fingerprint.json
    fold_all/checkpoint_best.pth      权重（374 MB）
    fold_all/debug.json              训练现场（GPU、torch、路径、全部超参）
    fold_all/training_log_*.txt      训练日志
code/nnUNetTrainer_noMirror.py       关闭 mirroring 的 trainer（3 行有效代码）
code/install_trainer.py              一键装进当前 nnU-Net
scripts/predict.sh                   推理（已带 --disable_tta）
scripts/train.sh                     重训命令
PROVENANCE.md                        训练溯源 / model card
```

## 怎么用（已验证）

```bash
python code/install_trainer.py
cp -r model/Dataset002_ToothFDI/nnUNetTrainer_noMirror__nnUNetPlans__2d $nnUNet_results/Dataset002_ToothFDI/
nnUNetv2_predict -i 输入 -o 输出 -d 2 -c 2d -tr nnUNetTrainer_noMirror -f all \
    -chk checkpoint_best.pth --disable_tta
```

实测：官方 nnUNetv2 2.6.2（CPU，torch 2.5.1），单张 482×1024 全景片 3–4 秒出 `case.png`。
不加 `--disable_tta` 会直接崩：`ValueError: max() arg is an empty sequence`（因为 checkpoint 里
`inference_allowed_mirroring_axes = ()`）。这条必须写进 README，否则别人跑不起来。

不想装 trainer 也可以：把目录名改成 `nnUNetTrainer__nnUNetPlans__2d`，用 `-tr nnUNetTrainer`，
同样要加 `--disable_tta`。

## 三个必须知道的事实

1. **只有一个模型**：`fold_all`，在 95 张训练集上训练，不是五折，也不是五折集成。日志里 0.997 的 Dice 是这 95 张的训练集自评，不能当性能指标。
2. **划分是 95/5，不是五折**：`dataset.json` 的 `numTraining` 为 95。另外 5 张是留出的验证集，没有参与这次训练，也没有用来挑选 `checkpoint_best.pth`。
3. **训练没跑完**：计划 1000 epoch，第 768 就停了，best 在 765。论文若写"训练 1000 epoch"，与日志矛盾。

## 与回复信的关系

回复里要写的是：95/5 划分、未做五折；released checkpoint 是 fold `all` 的单模型，只在 95 张训练集上拟合；52 FDI classes + background；mirroring disabled；推理必须 `--disable_tta`。权重、配置、命令和训练日志在本仓库。**不能写 five-fold cross-validation / five-fold ensemble**。
