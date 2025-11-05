# 使用範例

## 數據準備範例

### 1. 組織您的數據

將您的數據按以下結構組織：

```
data/processed/train/
├── images/
│   ├── patient001_cervical_lateral.png
│   ├── patient001_cervical_ap.png
│   ├── patient002_lumbar_lateral.png
│   └── ...
├── annotations.csv
└── reports.json
```

### 2. 創建annotations.csv

```python
import pandas as pd

data = {
    'image_id': ['001', '002', '003'],
    'image_filename': [
        'patient001_cervical_lateral.png',
        'patient002_lumbar_ap.png',
        'patient003_thoracic_lateral.png'
    ],
    'patient_id': ['P001', 'P002', 'P003'],
    'spine_region': ['cervical', 'lumbar', 'thoracic'],
    'pose': ['lateral', 'neutral', 'lateral'],
    'pathologies': [
        'degenerative_disc_disease,spinal_stenosis',
        'herniated_disc',
        'normal'
    ],
    'implants': [
        'none',
        'screws,rods',
        'none'
    ]
}

df = pd.DataFrame(data)
df.to_csv('data/processed/train/annotations.csv', index=False)
```

### 3. 創建reports.json

```python
import json

reports = {
    "001": {
        "findings": "Mild degenerative disc disease at C5-C6 with disc space narrowing. Moderate spinal canal stenosis at this level.",
        "impression": "Cervical spondylosis with spinal stenosis at C5-C6"
    },
    "002": {
        "findings": "Post-surgical changes with pedicle screw fixation at L4-L5. Disc herniation at L5-S1 level with nerve root impingement.",
        "impression": "Status post lumbar fusion. L5-S1 disc herniation"
    },
    "003": {
        "findings": "Normal alignment of thoracic spine. No acute fracture or dislocation. No significant degenerative changes.",
        "impression": "Normal thoracic spine"
    }
}

with open('data/processed/train/reports.json', 'w', encoding='utf-8') as f:
    json.dump(reports, f, indent=2, ensure_ascii=False)
```

## 完整訓練流程

### 步驟1：準備embeddings

```bash
# 準備訓練集embeddings
python scripts/prepare_embeddings.py \
    --reports_file data/processed/train/reports.json \
    --annotations_file data/processed/train/annotations.csv \
    --output_dir data/embeddings/train \
    --train_embedder

# 準備驗證集embeddings
python scripts/prepare_embeddings.py \
    --reports_file data/processed/val/reports.json \
    --output_dir data/embeddings/val

# 準備測試集embeddings
python scripts/prepare_embeddings.py \
    --reports_file data/processed/test/reports.json \
    --output_dir data/embeddings/test
```

### 步驟2：開始訓練

```bash
# 基礎訓練
python scripts/train.py \
    --config configs/train_config.yaml \
    --output_dir checkpoints/exp001

# 使用多GPU訓練
accelerate launch scripts/train.py \
    --config configs/train_config.yaml \
    --output_dir checkpoints/exp001_multi_gpu

# 從檢查點恢復訓練
python scripts/train.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/exp001/checkpoint_step_1000 \
    --output_dir checkpoints/exp001
```

### 步驟3：評估模型

```bash
# 在測試集上評估
python scripts/evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/exp001/best_model \
    --split test \
    --output_dir results/exp001

# 評估所有分割
for split in train val test; do
    python scripts/evaluate.py \
        --config configs/train_config.yaml \
        --checkpoint checkpoints/exp001/best_model \
        --split $split \
        --output_dir results/exp001
done
```

### 步驟4：推理

```bash
# 單張影像推理
python scripts/inference.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/exp001/best_model \
    --image data/test_images/case001.png \
    --generate_report \
    --output results/case001_result.json

# 批量推理
for image in data/test_images/*.png; do
    output="results/$(basename $image .png)_result.json"
    python scripts/inference.py \
        --config configs/train_config.yaml \
        --checkpoint checkpoints/exp001/best_model \
        --image "$image" \
        --generate_report \
        --output "$output"
done
```

## Python API使用範例

### 載入模型並推理

```python
import torch
from PIL import Image
from src.models.medgemma_model import MedGemmaSpineModel
from src.data.preprocessing import SpineImagePreprocessor

# 載入模型
model = MedGemmaSpineModel.from_pretrained(
    "checkpoints/best_model",
    num_regions=3,
    num_poses=4,
    num_pathologies=10,
    num_implants=6
)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 準備影像
preprocessor = SpineImagePreprocessor(
    image_size=(512, 512)
)
image = Image.open("test_image.png").convert('RGB')
image_tensor = preprocessor(image).unsqueeze(0).to(device)

# 準備文本輸入
tokenizer = model.tokenizer
prompt = "Analyze this spine X-ray image."
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)

# 推理
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        image_features=image_tensor
    )

# 獲取預測結果
region_pred = torch.argmax(outputs['region_logits'], dim=1).item()
pose_pred = torch.argmax(outputs['pose_logits'], dim=1).item()
pathology_probs = torch.sigmoid(outputs['pathology_logits'])[0]
implant_probs = torch.sigmoid(outputs['implant_logits'])[0]

print(f"Predicted region: {region_pred}")
print(f"Predicted pose: {pose_pred}")
print(f"Pathology probabilities: {pathology_probs}")
```

### 生成報告

```python
# 使用模型生成放射科報告
report = model.generate_report(
    prompt="Analyze this spine X-ray and provide a detailed radiology report:",
    image_features=image_tensor,
    max_length=512,
    temperature=0.7
)

print("Generated Report:")
print(report)
```

### 訓練自定義數據集

```python
from src.data.dataset import SpineXrayDataset
from src.training.trainer import SpineXrayTrainer
from torch.utils.data import DataLoader

# 創建數據集
train_dataset = SpineXrayDataset(
    data_dir="data/processed/train",
    split="train"
)

val_dataset = SpineXrayDataset(
    data_dir="data/processed/val",
    split="val"
)

# 創建dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4
)

# 訓練
trainer = SpineXrayTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config
)

trainer.train()
```

## 常見問題

### Q: 顯存不足怎麼辦？
A:
1. 減小batch_size
2. 增加gradient_accumulation_steps
3. 使用4-bit量化
4. 減小影像尺寸

### Q: 如何處理類別不平衡？
A: 數據集會自動計算類別權重，也可以手動調整任務權重。

### Q: 可以使用DICOM格式嗎？
A: 可以！數據集會自動處理DICOM格式。

### Q: 如何添加新的病變類別？
A: 在config.yaml中的pathologies列表添加新類別即可。
