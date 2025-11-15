# 脊椎測量功能使用範例

本文檔說明如何使用自動化脊椎測量功能，包括基準線檢測、椎體間距測量、Cobb角度等。

## 功能概覽

### 自動檢測的關鍵點
每個椎體檢測7個關鍵點：
1. **Superior Anterior** - 上終板前緣
2. **Superior Posterior** - 上終板後緣
3. **Inferior Anterior** - 下終板前緣
4. **Inferior Posterior** - 下終板後緣
5. **Left Pedicle** - 左側椎弓根
6. **Right Pedicle** - 右側椎弓根
7. **Spinous Process** - 棘突

### 自動計算的測量值

1. **椎體高度** (Vertebral Body Height)
   - 前緣高度 (Anterior height)
   - 後緣高度 (Posterior height)
   - 中線高度 (Mid height)
   - 楔形變化 (Wedging) - 用於檢測壓迫性骨折

2. **椎間盤高度** (Disc Height)
   - 前方椎間隙
   - 後方椎間隙
   - 中線椎間隙
   - 椎間盤楔形 (Disc wedging)

3. **脊椎排列** (Spinal Alignment)
   - 前緣線 (Anterior Vertebral Line, AVL)
   - 後緣線 (Posterior Vertebral Line, PVL)
   - 偏離量 (Deviation from ideal alignment)

4. **角度測量** (Angular Measurements)
   - Cobb角度 - 用於評估脊柱側彎
   - 前凸/後凸角度 (Lordosis/Kyphosis)

5. **滑脫評估** (Spondylolisthesis)
   - 滑脫距離 (mm)
   - 滑脫百分比 (%)
   - Meyerding分級 (Grade 1-4)

## 使用方法

### 1. 基本使用

```bash
# 對單張影像進行測量
python scripts/measure_spine.py \
    --image data/xrays/lumbar_ap.png \
    --region lumbar \
    --visualize \
    --output_dir results/measurements
```

### 2. 指定像素間距

如果知道影像的像素間距（從DICOM metadata或校準）：

```bash
python scripts/measure_spine.py \
    --image data/xrays/cervical_lateral.dcm \
    --region cervical \
    --pixel_spacing 0.143 \
    --visualize
```

### 3. 使用訓練好的關鍵點檢測器

```bash
python scripts/measure_spine.py \
    --image data/xrays/thoracic_ap.png \
    --checkpoint checkpoints/keypoint_detector/best_model.pt \
    --region thoracic \
    --visualize \
    --show_plots
```

## Python API 使用範例

### 範例 1：基本測量

```python
import numpy as np
from PIL import Image
from src.utils.measurements import (
    VertebraKeypoints,
    SpineMeasurements,
    create_measurement_report
)

# 定義椎體關鍵點（正規化坐標 0-1）
vertebra1 = VertebraKeypoints(
    superior_anterior=(0.3, 0.2),
    superior_posterior=(0.7, 0.2),
    inferior_anterior=(0.3, 0.3),
    inferior_posterior=(0.7, 0.3)
)

vertebra2 = VertebraKeypoints(
    superior_anterior=(0.3, 0.35),
    superior_posterior=(0.7, 0.35),
    inferior_anterior=(0.3, 0.45),
    inferior_posterior=(0.7, 0.45)
)

vertebrae = [vertebra1, vertebra2]

# 計算測量值
measurements = SpineMeasurements(pixel_spacing=0.5)  # 0.5 mm/pixel

# 椎體高度
height = measurements.compute_vertebral_body_height(vertebra1, 'anterior')
print(f"Vertebral height: {height:.1f} mm")

# 椎間盤高度
disc_height = measurements.compute_disc_height(vertebra1, vertebra2, 'mid')
print(f"Disc height: {disc_height:.1f} mm")

# Cobb角度
cobb_angle = measurements.compute_cobb_angle(vertebra1, vertebra2)
print(f"Cobb angle: {cobb_angle:.1f}°")
```

### 範例 2：完整測量報告

```python
from src.utils.measurements import create_measurement_report
import json

# 創建完整報告
report = create_measurement_report(
    vertebrae,
    pixel_spacing=0.5,
    region="lumbar"
)

# 輸出JSON報告
print(json.dumps(report, indent=2))

# 報告內容包括：
# - num_vertebrae: 檢測到的椎體數量
# - vertebral_heights: 每個椎體的高度測量
# - disc_heights: 椎間盤高度
# - alignment: 排列偏差
# - curvature: 彎曲角度
# - spondylolisthesis: 滑脫檢測
```

### 範例 3：可視化

```python
import cv2
from src.utils.visualization import SpineVisualizer

# 載入影像
image = cv2.imread('lumbar_xray.png')

# 創建可視化器
visualizer = SpineVisualizer(pixel_spacing=0.5)

# 繪製關鍵點
annotated = visualizer.draw_keypoints(image, vertebrae, show_labels=True)

# 繪製排列線
annotated = visualizer.draw_alignment_lines(annotated, vertebrae)

# 繪製測量值
annotated = visualizer.draw_measurements(
    annotated,
    vertebrae,
    show_heights=True,
    show_disc_spaces=True,
    show_cobb_angle=True
)

# 保存結果
cv2.imwrite('annotated_spine.png', annotated)

# 創建測量圖表
fig = visualizer.create_measurement_figure(vertebrae, region="lumbar")
fig.savefig('measurements_plot.png', dpi=300)
```

### 範例 4：滑脫檢測

```python
# 檢測椎體滑脫
slip = measurements.compute_spondylolisthesis(vertebra1, vertebra2)

print(f"Translation: {slip['translation_mm']:.1f} mm")
print(f"Translation %: {slip['translation_percent']:.1f}%")
print(f"Meyerding Grade: {slip['grade']}")

# 判斷臨床意義
if slip['grade'] >= 2:
    print("⚠️ Significant spondylolisthesis detected!")
```

### 範例 5：排列分析

```python
# 計算排列偏差
alignment = measurements.compute_alignment_deviation(vertebrae)

print("Anterior deviations (mm):", alignment['anterior_deviations'])
print("Posterior deviations (mm):", alignment['posterior_deviations'])

# 檢查是否有明顯偏離
max_deviation = np.max(np.abs(alignment['anterior_deviations']))
if max_deviation > 3.0:
    print(f"⚠️ Significant alignment deviation: {max_deviation:.1f} mm")
```

## 測量報告範例

執行測量腳本後，會生成JSON格式的報告：

```json
{
  "region": "lumbar",
  "num_vertebrae": 5,
  "vertebral_heights": [
    {
      "vertebra_index": 0,
      "anterior_mm": 28.5,
      "posterior_mm": 27.8,
      "mid_mm": 28.15,
      "wedging_mm": 0.7
    },
    ...
  ],
  "disc_heights": [
    {
      "anterior_mm": 12.3,
      "posterior_mm": 11.8,
      "mid_mm": 12.05,
      "wedging_mm": 0.5,
      "wedging_angle": 1.2
    },
    ...
  ],
  "alignment": {
    "anterior_deviations_mm": [0.2, -0.3, 0.1, ...],
    "posterior_deviations_mm": [0.1, -0.2, 0.3, ...],
    "max_anterior_deviation_mm": 0.3,
    "max_posterior_deviation_mm": 0.3
  },
  "curvature": {
    "cobb_angle": 8.5
  },
  "spondylolisthesis": []
}
```

## 臨床應用

### 1. 椎間盤退化評估

```python
# 比較相鄰椎間盤高度
disc_heights = measurements.compute_intervertebral_distances(vertebrae)

for i, disc in enumerate(disc_heights):
    if i > 0:
        # 計算高度損失百分比
        height_loss = (disc_heights[i-1]['mid_mm'] - disc['mid_mm']) / disc_heights[i-1]['mid_mm'] * 100
        if height_loss > 30:
            print(f"⚠️ Disc {i}: Significant height loss ({height_loss:.1f}%)")
```

### 2. 壓迫性骨折檢測

```python
# 檢查椎體楔形變化
for vh in vertebral_heights:
    wedging_percent = abs(vh['wedging_mm']) / vh['mid_mm'] * 100
    if wedging_percent > 20:
        print(f"⚠️ Vertebra {vh['vertebra_index']}: Possible compression fracture")
        print(f"   Wedging: {vh['wedging_mm']:.1f} mm ({wedging_percent:.1f}%)")
```

### 3. 脊柱側彎評估

```python
# Cobb角度分級
cobb = measurements.compute_cobb_angle(vertebrae[0], vertebrae[-1])

if cobb < 10:
    severity = "正常或輕微"
elif cobb < 25:
    severity = "輕度側彎"
elif cobb < 40:
    severity = "中度側彎"
else:
    severity = "重度側彎"

print(f"Cobb角度: {cobb:.1f}° - {severity}")
```

## 配置選項

在 `configs/measurement_config.yaml` 中可調整：

```yaml
measurements:
  pixel_spacing: 0.5  # 像素間距 (mm/pixel)
  min_disc_height_mm: 2.0  # 最小正常椎間盤高度
  max_cobb_angle_normal: 10.0  # 正常Cobb角度閾值
  spondylolisthesis_threshold: 25.0  # 滑脫百分比閾值

visualization:
  show_labels: true  # 顯示椎體標籤
  show_measurements: true  # 顯示測量值
  heatmap_alpha: 0.5  # 熱圖透明度
```

## 注意事項

1. **像素間距校準**：準確的測量需要正確的像素間距。從DICOM文件可自動提取。

2. **影像質量**：影像質量直接影響關鍵點檢測準確性。建議：
   - 良好的曝光和對比度
   - 清晰的椎體邊界
   - 適當的姿勢（正位或側位）

3. **關鍵點檢測**：
   - 需要訓練好的關鍵點檢測器
   - Demo模式僅供展示，不可用於臨床

4. **臨床使用**：
   - 測量結果僅供參考
   - 需由專業醫師驗證
   - 不能替代人工測量

## 下一步

1. 訓練關鍵點檢測器（需標註數據）
2. 整合到完整的診斷流程
3. 添加更多測量指標
4. 開發自動報告生成

## 相關資源

- [脊椎測量配置](../configs/measurement_config.yaml)
- [測量腳本](../scripts/measure_spine.py)
- [API文檔](../src/utils/measurements.py)
- [可視化工具](../src/utils/visualization.py)
