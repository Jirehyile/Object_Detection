# CSFPR-RTDETR 论文复现流程

本项目用于复现论文 **CSFPR-RTDETR: Real-Time Small Object Detection Network for UAV Images Based on Cross-Spatial-Frequency Domain and Position Relation**，并为后续模型改进和论文研究建立可信基线。

本文默认已经安装 Python、PyTorch 和对应的 CUDA 环境。复现应按照以下顺序进行：

标记说明：

- **[必须]**：进入创新阶段前必须完成；
- **[投稿前补充]**：可以在创新阶段同步开展，但正式投稿前应完成。

```text
环境确认 → 数据准备 → 官方权重验证 → 冒烟训练 → 固定正式配置
→ 训练基线 → 训练完整模型 → 关键消融 → 效率测试
→ 结果分析与稳定性验证 → 整理复现材料
```

## 阶段一 环境与项目确认 [必须]

### 需要完成

- 安装项目依赖；
- 确认 PyTorch 能识别 CUDA；
- 确认导入的是项目内置的 Ultralytics；
- 确认官方权重可以加载。

### 实施方式

```bash
python -m pip install -r requirements.txt
python -m pip install grad-cam timm einops dill
```

```bash
python -c "import torch, ultralytics; print(torch.__version__); print(torch.cuda.is_available()); print(ultralytics.__version__); print(ultralytics.__file__)"
```

### 完成标志

- `torch.cuda.is_available()` 返回 `True`；
- `ultralytics.__file__` 指向本项目目录；
- `weights/visdrone.pt` 可以正常加载，无缺失模块错误。

## 阶段二 数据集准备与校验 [必须]

### 需要完成

- 下载 VisDrone2019 DET 的 trainset、valset 和 testset-dev；
- 将 VisDrone 原始标注转换为 YOLO 格式；
- 配置 `dataset/visdrone.yaml`；
- 检查图片数量、标签数量、类别编号和坐标范围。

### 实施方式

推荐目录结构：

```text
Data/
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

执行标注转换：

```bash
python dataset/visdrone_to_yolo.py --data-root /path/to/Data
```

转换规则：

- 忽略不参与十类检测的标注；
- 将 VisDrone 类别 `1～10` 映射为 YOLO 类别 `0～9`；
- 将边界框转换为归一化 YOLO 坐标；
- 保留原始 `annotations`，转换结果写入 `labels`。

修改 `dataset/visdrone.yaml`：

```yaml
path: /path/to/Data
train: VisDrone2019-DET-train
val: VisDrone2019-DET-val
test: VisDrone2019-DET-test-dev
```

### 完成标志

- train、val、test-dev 分别包含 6471、548、1610 张图片；
- 每张图片都有对应的 YOLO 标签文件；
- 类别编号只包含 `0～9`；
- 归一化坐标位于 `0～1`；
- Ultralytics 扫描数据时没有损坏图片或标签错误。

## 阶段三 官方权重验证 [必须]

### 需要完成

- 使用官方权重完成单张图片推理；
- 使用 `visdrone.pt` 在 test-dev 上完成完整评测；
- 对比论文公开指标。

### 实施方式

单张图片推理：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('weights/visdrone.pt'); model.predict(source='test_images/test.jpg', imgsz=640, device=0, save=True)"
```

完整评测：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('weights/visdrone.pt'); model.val(data='dataset/visdrone.yaml', split='test', imgsz=640, batch=4, device=0)"
```

论文 VisDrone 参考值：

```text
Precision:  58.6%
Recall:     43.3%
mAP50:      42.3%
mAP50:95:   24.9%
Params:     14.09 M
GFLOPs:     63.9
```

### 完成标志

- 推理结果中的类别和边界框显示正常；
- test-dev 评测可以完整结束；
- 官方权重评测结果与论文指标基本一致；
- 模型参数量和 GFLOPs 与论文一致。

该阶段通过后，才能确认项目代码、数据转换和评测协议正确。

## 阶段四 基线与完整模型冒烟训练 [必须]

### 需要完成

- RT-DETR 基线训练 1 epoch；
- CSFPR-RTDETR 完整模型训练 1 epoch；
- 检查显存、loss、验证流程和权重保存。

### 实施方式

RT-DETR 基线：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/rtdetr.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=1, workers=0, device=0, project='runs/smoke', name='rtdetr')"
```

CSFPR-RTDETR：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=1, batch=1, workers=0, device=0, project='runs/smoke', name='csfpr-rtdetr')"
```

### 完成标志

- 两个模型都能完成训练和验证；
- loss 为有限数值，没有 `nan`；
- 没有显存溢出或数据加载错误；
- 成功生成 `best.pt`、`last.pt`、`results.csv` 和训练曲线；
- 记录一个 epoch 的耗时和显存占用。

冒烟训练只用于检查流程，不用于判断最终精度

## 阶段五 固定正式实验配置 [必须]

### 需要完成

- 根据冒烟结果确定 batch、workers 和混合精度设置；
- 固定输入尺寸、优化器、学习率、训练轮数和随机种子；
- 确保基线和完整模型使用相同实验条件。

### 实施方式

正式实验至少记录：

```text
Python、PyTorch、CUDA、GPU
模型配置文件
数据集版本和划分
imgsz、epochs、batch、workers
optimizer、lr0、weight_decay
AMP 设置和随机种子
```

显存不足时可以减小 batch，但必须保存实际配置。不要在未记录的情况下修改输入尺寸或训练策略。

示例中的单行命令使用 `workers=0` 以兼容 Windows。正式训练若需提高数据加载速度，应使用带有 `if __name__ == '__main__':` 的独立 Python 脚本，再逐步提高 workers。

### 完成标志

- 形成一份固定的正式训练配置；
- 配置能够在当前硬件上稳定运行；
- RT-DETR 和 CSFPR-RTDETR 除模型结构外保持相同条件。

## 阶段六 完整训练 RT-DETR 基线 [必须]

### 需要完成

- 按正式配置训练 RT-DETR；
- 保存最优权重、最终权重、日志和曲线；
- 在 test-dev 上评测本地基线。

### 实施方式

将冒烟训练命令中的 `epochs=1` 改为论文训练轮数，例如：

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/rtdetr.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=300, batch=4, workers=0, device=0, project='runs/train', name='rtdetr-visdrone')"
```

论文中的 RT-DETR 基线参考值：

```text
mAP50:    37.7%
mAP50:95: 21.8%
```

### 完成标志

- 完成全部训练轮次；
- loss 和验证指标变化合理；
- 得到可用的本地 RT-DETR 基线权重；
- 本地基线结果与论文基线处于合理范围；
- 训练配置、日志和评测结果完整保存。

## 阶段七 完整训练 CSFPR-RTDETR [必须]

### 需要完成

- 使用与基线相同的正式配置训练完整模型；
- 在 test-dev 上评测；
- 与本地 RT-DETR 基线公平比较。

### 实施方式

```bash
python -c "from ultralytics import RTDETR; model=RTDETR('ultralytics/cfg/modelY/CSFPR-RTDETR.yaml'); model.train(data='dataset/visdrone.yaml', imgsz=640, epochs=300, batch=4, workers=0, device=0, project='runs/train', name='csfpr-rtdetr-visdrone')"
```

### 完成标志

- 完成全部训练轮次并得到最优权重；
- 本地完整模型的指标接近论文报告结果；
- 完整模型在相同条件下明显优于本地 RT-DETR 基线；
- 精度提升方向与论文一致。

完整模型应主要与本地重新训练的基线比较，不能只与论文表格中的数值比较。

## 阶段八 关键模块消融复现 [必须]

### 需要完成

至少验证以下配置：

1. RT-DETR 基线；
2. RT-DETR + CSFH；
3. RT-DETR + Position Relation；
4. RT-DETR + P2/MHE-Fusion；
5. 完整 CSFPR-RTDETR。

### 实施方式

- 为每种结构建立独立模型配置；
- 保持训练和评测参数一致；
- 每个实验独立保存权重、日志和结果；
- 汇总 Precision、Recall、mAP50、mAP50:95、参数量和 GFLOPs。

### 完成标志

- 关键模块能够独立训练和评测；
- 各模块带来的变化方向与论文消融基本一致；
- 能解释单个模块的收益、代价以及组合效果。

## 阶段九 精度与效率统一测试 [必须]

### 需要完成

对基线和完整模型统一测试：

- Precision、Recall、mAP50、mAP50:95；
- 参数量和 GFLOPs；
- 推理时延或 FPS；
- 峰值显存。

### 实施方式

速度测试应统一以下条件：

```text
同一 GPU
相同输入尺寸和 batch
相同 FP32/FP16 设置
相同测试数据
相同预热次数和计时次数
明确是否包含前处理和后处理
```

### 完成标志

- 获得可重复的精度与效率对比表；
- 参数量和 GFLOPs 与模型配置一致；
- 能说明完整模型相对基线的精度收益和速度代价。

## 阶段十 结果分析与稳定性验证 [投稿前补充]

### 需要完成

- 分析正确检测、漏检和误检案例；
- 重点观察密集遮挡、极小目标和复杂背景；
- 对核心模型进行重复实验；
- 条件允许时扩展到 AI-TOD 或 HIT-UAV。

### 实施方式

- 保存基线与完整模型的对比可视化；
- 对关键正式实验使用多个随机种子；
- 报告均值和波动，而非只报告最佳一次结果；
- 第二数据集沿用相同的数据校验和评测流程。

### 完成标志

- 能明确说明模型在哪些场景有效、在哪些场景仍存在问题；
- 核心提升在重复实验中保持稳定；
- 后续创新问题能够由复现实验结果支持，而不是仅凭主观判断。

## 阶段十一 整理复现材料 [必须]

### 需要完成

为每项正式实验保存：

```text
代码版本和模型配置
环境与依赖版本
数据处理脚本和数据配置
完整训练参数
训练日志和曲线
best.pt 和 last.pt
测试指标
速度与显存结果
典型可视化和失败案例
```

### 完成标志

- 可以根据记录重新运行任一实验；
- 不同模型之间的实验条件可以核对；
- 结果表中的每个数字都能追溯到具体配置、日志和权重。

## 完整复现成功判据

同时满足以下条件，可认为复现阶段完成：

- 官方权重在 VisDrone 上的评测结果与论文一致；
- 本地 RT-DETR 基线完成完整训练并得到合理指标；
- 本地 CSFPR-RTDETR 完成完整训练并稳定优于本地基线；
- 关键模块消融趋势与论文结论基本一致；
- 参数量、GFLOPs、速度和显存测试方法统一；
- 核心结果具有基本重复性；
- 环境、配置、日志、权重和结果均已归档。

完成以上复现后，再进入创新阶段。后续改进模型应同时与本地 RT-DETR 基线和本地 CSFPR-RTDETR 原模型比较。
