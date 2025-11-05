# MedGemma 27B 脊椎X光影像微調專案

這個專案使用**MedGemma 27B**模型進行脊椎X光影像的多任務學習微調，能夠進行：

1. ✅ **脊椎區域識別** - 辨識cervical（頸椎）、thoracic（胸椎）、lumbar（腰椎）
2. ✅ **姿勢識別** - 辨識flexion、extension、oblique pose等不同姿勢
3. ✅ **骨科診斷** - 診斷常見脊椎病變（退化性病變、椎間盤突出、壓迫性骨折等）
4. ✅ **神經孔系統分析** - 分析神經孔狹窄等問題
5. ✅ **植入物判斷** - 識別screw、rod、cage等植入物
6. ✅ **報告生成** - 自動生成放射科報告
7. ✅ **語意Embedding** - 針對脊椎報告進行語意embedding訓練

## 📁 專案結構

```
medgamme_finetuning_x-ray_spine/
├── configs/
│   ├── train_config.yaml          # 訓練配置
│   └── medical_terms.txt          # 醫學術語詞典
├── data/
│   ├── raw/                       # 原始數據
│   ├── processed/                 # 處理後的數據
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── embeddings/                # 報告embeddings
├── src/
│   ├── data/
│   │   ├── dataset.py            # 數據集類別
│   │   ├── preprocessing.py      # 影像預處理
│   │   └── augmentation.py       # 數據增強
│   ├── models/
│   │   └── medgemma_model.py     # MedGemma模型
│   ├── training/
│   │   └── trainer.py            # 訓練器
│   └── utils/
│       ├── embeddings.py         # 文本embedding
│       └── metrics.py            # 評估指標
├── scripts/
│   ├── train.py                  # 訓練腳本
│   ├── evaluate.py               # 評估腳本
│   ├── inference.py              # 推理腳本
│   └── prepare_embeddings.py     # Embedding準備腳本
├── notebooks/                     # Jupyter notebooks
├── checkpoints/                   # 模型檢查點
├── logs/                         # 訓練日誌
└── requirements.txt              # 依賴套件

```

## 🚀 快速開始

### 1. 環境安裝

```bash
# 克隆專案
git clone <repository-url>
cd medgamme_finetuning_x-ray_spine

# 創建虛擬環境（建議使用Python 3.9+）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 安裝spaCy模型（用於NLP處理）
python -m spacy download en_core_web_sm
```

### 2. 數據準備

您的數據應該按以下格式組織：

#### 數據格式

**annotations.csv** - 影像標註文件：
```csv
image_id,image_filename,patient_id,spine_region,pose,pathologies,implants
001,patient001_cervical_lat.png,P001,cervical,lateral,"degenerative_disc_disease,spinal_stenosis","none"
002,patient002_lumbar_ap.png,P002,lumbar,neutral,"herniated_disc","screws,rods"
```

**reports.json** - 放射科報告：
```json
{
  "001": {
    "findings": "Mild degenerative changes at C5-C6 level with disc space narrowing...",
    "impression": "Cervical spondylosis"
  },
  "002": {
    "findings": "Post-surgical changes with pedicle screw fixation at L4-L5...",
    "impression": "Status post lumbar fusion"
  }
}
```

**影像文件** - 支援格式：
- PNG, JPG（建議使用PNG保留更多細節）
- DICOM (.dcm)

### 3. 準備報告Embeddings（可選但建議）

```bash
python scripts/prepare_embeddings.py \
    --reports_file data/processed/train/reports.json \
    --annotations_file data/processed/train/annotations.csv \
    --output_dir data/embeddings \
    --train_embedder
```

這會：
- 建立醫學詞彙表
- 訓練領域自適應的embedding模型
- 生成所有報告的embeddings

### 4. 訓練模型

```bash
python scripts/train.py \
    --config configs/train_config.yaml \
    --output_dir checkpoints/exp001
```

使用多GPU訓練：
```bash
accelerate config  # 首次配置
accelerate launch scripts/train.py --config configs/train_config.yaml
```

### 5. 評估模型

```bash
python scripts/evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/best_model \
    --split test \
    --output_dir results
```

### 6. 推理

```bash
python scripts/inference.py \
    --config configs/train_config.yaml \
    --checkpoint checkpoints/best_model \
    --image path/to/xray.png \
    --generate_report \
    --output results/inference_result.json
```

## ⚙️ 配置說明

### 主要配置項（train_config.yaml）

#### 模型配置
```yaml
model:
  name: "google/medgemma-27b"
  use_lora: true              # 使用LoRA高效微調
  lora_r: 16                  # LoRA rank
  lora_alpha: 32
  load_in_4bit: true          # 4-bit量化（節省顯存）
```

#### 數據配置
```yaml
data:
  image_size: [512, 512]      # 影像大小
  batch_size: 4               # 批次大小
  use_augmentation: true      # 數據增強
```

#### 訓練配置
```yaml
training:
  num_epochs: 50
  learning_rate: 2.0e-4
  gradient_accumulation_steps: 4  # 梯度累積
  task_weights:                   # 各任務權重
    region_classification: 1.0
    pose_classification: 1.0
    pathology_classification: 2.0
    implant_detection: 1.5
```

## 🎯 模型能力

### 1. 脊椎區域分類
- Cervical（頸椎）C1-C7
- Thoracic（胸椎）T1-T12
- Lumbar（腰椎）L1-L5

### 2. 姿勢分類
- Neutral（中立位）
- Flexion（屈曲）
- Extension（伸展）
- Oblique（斜位）

### 3. 病變檢測（多標籤）
- Normal（正常）
- Degenerative disc disease（退化性椎間盤病變）
- Herniated disc（椎間盤突出）
- Spinal stenosis（脊椎狹窄）
- Spondylolisthesis（脊椎滑脫）
- Compression fracture（壓迫性骨折）
- Scoliosis（脊柱側彎）
- Ankylosing spondylitis（僵直性脊椎炎）
- Infection（感染）
- Tumor（腫瘤）

### 4. 植入物檢測（多標籤）
- Screws（螺釘）
- Rods（鋼棒）
- Cage（融合器）
- Artificial disc（人工椎間盤）
- Bone graft（骨移植）

## 📊 性能指標

模型會計算以下指標：
- Accuracy（準確率）
- Precision/Recall/F1（精確率/召回率/F1分數）
- AUC-ROC
- 混淆矩陣
- 每個類別的詳細指標

## 💡 技術特點

### 1. LoRA高效微調
使用LoRA（Low-Rank Adaptation）技術，只訓練少量參數：
- 節省顯存：4-bit量化 + LoRA可在單張24GB GPU上訓練27B模型
- 訓練速度快：只更新約1-2%的參數
- 效果好：在醫學影像任務上與全量微調相當

### 2. 多任務學習
同時學習4個相關任務，提升模型泛化能力：
- 共享底層特徵表示
- 任務間互相促進學習
- 可配置不同任務的權重

### 3. 醫學影像專用預處理
- CLAHE（對比度限制自適應直方圖均衡化）
- 骨骼增強
- DICOM支援與窗位調整

### 4. 領域自適應Embedding
- 針對脊椎報告訓練的語意embedding
- 醫學詞彙提取
- 支援報告相似度計算

### 5. 數據增強策略
保守的醫學影像增強：
- 小角度旋轉（±10度）
- 輕微亮度/對比度調整
- MixUp/CutMix（可選）

## 🔧 故障排除

### 顯存不足
```yaml
# 在train_config.yaml中調整：
data:
  batch_size: 2                    # 減小批次大小
training:
  gradient_accumulation_steps: 8   # 增加梯度累積
model:
  load_in_4bit: true              # 使用4-bit量化
```

### 訓練不穩定
```yaml
training:
  learning_rate: 1.0e-4           # 降低學習率
  warmup_steps: 1000              # 增加warmup步數
  max_grad_norm: 0.5              # 降低梯度裁剪閾值
```

### 數據不平衡
模型會自動計算類別權重，也可手動調整：
```python
# 在dataset.py中
weights = dataset.get_class_weights(task='pathology')
```

## 📈 監控訓練

使用Weights & Biases監控：
```yaml
wandb:
  enabled: true
  project: "medgemma-spine-xray"
```

訪問 https://wandb.ai 查看訓練進度、指標曲線等。

## 🤝 貢獻

歡迎提交Issue和Pull Request！

## 📄 授權

[請根據您的需求添加授權信息]

## 📚 參考資料

- [MedGemma論文](https://arxiv.org/abs/...)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [脊椎影像學指南](...)

## 📧 聯繫方式

如有問題，請聯繫：[您的聯繫方式]

---

**注意事項**：
1. 此模型僅供研究使用，不能替代專業醫療診斷
2. 使用前請確保有適當的醫療數據使用授權
3. 建議由專業放射科醫師驗證模型輸出
