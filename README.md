# CSFPR-RTDETR 论文复现

本项目用于复现论文 **CSFPR-RTDETR: Real-Time Small Object Detection Network for UAV Images Based on Cross-Spatial-Frequency Domain and Position Relation**。

以下流程默认已经安装 Python、PyTorch 和对应的 CUDA 环境。

## 1. 安装依赖

进入项目根目录：

```bash
cd CSFPR-RTDETR
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install grad-cam timm einops dill
```

检查环境和项目包：

```bash
python -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"
```

## 2. 官方权重推理

项目提供以下权重：

```text
weights/visdrone.pt
weights/HITUAV.pt
weights/AITOD.pt
```

将测试图片放入 `test_images/test.jpg`，使用 VisDrone 权重进行推理：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('weights/visdrone.pt'); model.predict(source='test_images/test.jpg', imgsz=640, conf=0.25, device=0, save=True, project='runs/predict', name='visdrone_test')"
```

结果保存在 `runs/predict/visdrone_test/`。

## 3. 准备 VisDrone 数据集

下载 **VisDrone2019 Task 1: Object Detection in Images** 的以下数据：

- trainset；
- valset；
- testset-dev。

建议目录结构：

```text
datasets/visdrone2019/
├── VisDrone2019-DET-train/
│   ├── images/
│   ├── annotations/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   ├── annotations/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    ├── images/
    ├── annotations/
    └── labels/
```

VisDrone 原始标注不是 YOLO 格式，训练前需要完成以下转换：

- 忽略类别 `0`；
- 将类别 `1～10` 映射为 `0～9`；
- 将边界框转换为归一化的 YOLO 格式；
- 保留原始 `annotations/`，将转换结果写入 `labels/`。

## 4. 配置数据路径

修改 `dataset/visdrone.yaml` 中的 `path`：

```yaml
path: /path/to/datasets/visdrone2019
train: VisDrone2019-DET-train
val: VisDrone2019-DET-val
test: VisDrone2019-DET-test-dev
```

Windows 可以使用：

```yaml
path: D:/datasets/visdrone2019
```

## 5. 评测官方权重

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('weights/visdrone.pt'); model.val(data='dataset/visdrone.yaml', split='test', imgsz=640, batch=4, workers=4, device=0)"
```

论文在 VisDrone 上的主要参考指标：

```text
mAP50:    42.3%
mAP50:95: 24.9%
```

如果显存不足，可以减小 `batch`，但应记录与论文设置的差异。

## 6. 冒烟训练

正式训练前，先分别训练 1 epoch，检查数据读取、loss、显存占用和结果保存是否正常。

RT-DETR 基线：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/rtdetr.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=4, workers=4, device=0, project='runs/smoke', name='rtdetr')"
```

CSFPR-RTDETR：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=4, workers=4, device=0, project='runs/smoke', name='csfpr-rtdetr')"
```

## 7. 完整训练

冒烟训练通过后，将 `epochs` 设置为 300：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/rtdetr.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=300, batch=4, workers=4, device=0, project='runs/train', name='rtdetr-visdrone')"
```

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=300, batch=4, workers=4, device=0, project='runs/train', name='csfpr-rtdetr-visdrone')"
```

训练顺序为：

1. RT-DETR 基线；
2. CSFPR-RTDETR 完整模型；
3. CSFH、Position Relation、P2 和 MHE-Fusion 等关键模块消融；
4. 在 VisDrone 复现稳定后，再扩展到 AI-TOD 和 HIT-UAV。

## 8. 注意事项

- 基线和完整模型应使用相同的数据划分、环境和训练参数；
- 输入尺寸保持为 `640×640`；
- 显存不足时可将 `batch` 从 4 降为 2 或 1；
- 不建议直接运行原始 `tools/train.py`，该脚本会连续训练三个数据集；
- 保存每次实验的配置、日志、权重和评测结果；
- 复现时重点比较本地基线与完整模型的相对提升趋势。
