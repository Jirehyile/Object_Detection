# CSFPR-RTDETR 论文复现

本项目用于复现论文 **CSFPR-RTDETR: Real-Time Small Object Detection Network for UAV Images Based on Cross-Spatial-Frequency Domain and Position Relation**。

总体顺序：**官方权重验证 → RT-DETR 基线 → CSFPR-RTDETR 完整模型 → 关键模块消融**。

## 1. 当前进度

已完成：

- 项目环境和 CUDA 可用；
- 项目内置 Ultralytics 8.0.201 可正常导入；
- `weights/visdrone.pt` 可正常加载；
- 已完成单张图片 GPU 推理。

下一步：下载并转换 VisDrone 数据集，使用官方权重进行完整评测。

## 2. 环境

作者仓库推荐：Python 3.11、PyTorch 2.1.0、Torchvision 0.16.0、CUDA 12.1。

除 `requirements.txt` 外，还需安装遗漏的依赖：

```powershell
C:\Users\l\.conda\envs\pytorch310\python.exe(修改为你的配置路径) -m pip install grad-cam timm einops dill
```

所有命令默认在项目根目录执行：

```powershell
cd D:\论文复现\CSFPR-RTDETR
```

## 3. 官方权重推理

将测试图片放在 `test_images/test.jpg`，然后执行：

```powershell
C:\Users\l\.conda\envs\pytorch310\python.exe -c(修改为你的配置路径) "from ultralytics import RTDETR; m=RTDETR('weights/visdrone.pt'); m.predict(source='test_images/test.jpg', imgsz=640, conf=0.25, device=0, save=True, project='runs/predict', name='visdrone_test')"
```

结果保存在 `runs/predict/visdrone_test/`。

## 4. VisDrone 数据准备

下载 **VisDrone2019 Task 1: Object Detection in Images** 的 trainset、valset 和 testset-dev。

建议目录：

```text
D:\论文复现\dataest\visdrone2019\
├── VisDrone2019-DET-train\
├── VisDrone2019-DET-val\
└── VisDrone2019-DET-test-dev\
```

官方下载的 `annotations` 不是 YOLO 格式，训练前必须：

- 忽略类别 `0`；
- 将类别 `1～10` 转为 `0～9`；
- 将边界框转换为归一化 YOLO 格式；
- 保留原始 `annotations/`，转换结果写入 `labels/`。

转换完成后修改 `dataset/visdrone.yaml`：

```yaml
path: D:/论文复现/dataest/visdrone2019
train: VisDrone2019-DET-train
val: VisDrone2019-DET-val
test: VisDrone2019-DET-test-dev
```

## 5. 官方权重评测

```powershell
C:\Users\l\.conda\envs\pytorch310\python.exe(修改为你的配置路径) -c "from ultralytics import RTDETR; m=RTDETR('weights/visdrone.pt'); m.val(data='dataset/visdrone.yaml', split='test', imgsz=640, batch=2, workers=2, device=0)"
```

论文 VisDrone 参考指标：`mAP50=42.3%`，`mAP50:95=24.9%`。

## 6. 训练流程

正式训练前，RT-DETR 基线和 CSFPR-RTDETR 都先运行 1 epoch，确认数据、loss 和显存正常。

RT-DETR 基线：

```powershell
C:\Users\l\.conda\envs\pytorch310\python.exe(修改为你的配置路径) -c "from ultralytics import RTDETR; m=RTDETR('ultralytics/cfg/modelY/rtdetr.yaml'); m.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=2, workers=2, device=0, project='runs/smoke', name='rtdetr')"
```

CSFPR-RTDETR：

```powershell
C:\Users\l\.conda\envs\pytorch310\python.exe(修改为你的配置路径) -c "from ultralytics import RTDETR; m=RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml'); m.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=2, workers=2, device=0, project='runs/smoke', name='csfpr')"
```

冒烟训练通过后，将 `epochs=1` 改为 `epochs=300`，先训练基线，再训练完整模型。

## 7. 注意事项

- 8 GB 显存先使用 `batch=2`，爆显存时降为 `batch=1`；
- 输入尺寸保持 `640×640`；
- 基线与完整模型使用相同环境、数据划分和训练参数；
- 不要直接运行原始 `tools/train.py`，它会连续训练三个数据集；
- 先复现 VisDrone，再考虑 AI-TOD 和 HIT-UAV；
- 完整模型跑通后，再做 CSFH、Position Relation、P2 和 MHE-Fusion 消融；
- 保存训练配置、日志、权重、曲线、显存和速度结果。

复现重点是官方权重评测可靠、基线正确，并且完整模型相对基线的提升趋势与论文一致。
