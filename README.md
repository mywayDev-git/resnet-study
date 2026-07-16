# 🧪 CIFAR-100 실험 히스토리 및 분석 리포트

```
layers.append(nn.Linear(num_ftrs, 1024))
layers.append(nn.BatchNorm1d(1024))
layers.append(nn.ReLU(inplace=True))
layers.append(nn.Dropout(0.3))
layers.append(nn.Linear(1024, 512))
layers.append(nn.BatchNorm1d(512))
layers.append(nn.ReLU(inplace=True))
layers.append(nn.Dropout(0.15))
layers.append(nn.Linear(512, 100))
model.fc = nn.Sequential(*layers)
```

> **총 실험 개수:** 14개 | **업데이트:** 2026-07-15 15:42

## 📊 전체 요약 (주요 변경점 위주)
| 실험 ID                |   정확도(%) | 주요 파라미터 변화                      | 옵티마이저   |   학습률 |
|:-----------------------|------------:|:----------------------------------------|:-------------|---------:|
| 01_baseline            |       81.83 | -                                       | sgd          |  0.001   |
| 02_high_lr             |       77.49 | lr(0.001→0.1)                           | sgd          |  0.1     |
| 03_adam_standard       |       80.94 | optimizer(sgd→adam)<br>lr(0.001→0.0001) | adam         |  0.0001  |
| 04_freeze_fc           |       61.31 | freeze_level(63→1)                      | sgd          |  0.001   |
| 05_freeze_partial_deep |       78.57 | freeze_level(63→7)                      | sgd          |  0.001   |
| 06_freeze_bottom_up    |       69.31 | freeze_level(63→33)                     | sgd          |  0.001   |
| 07_high_dropout        |       81.83 | drop_out(0.3→0.5)                       | sgd          |  0.001   |
| 08_no_smoothing        |       81.55 | label_smoothing(0.1→0.0)                | sgd          |  0.001   |
| 09_heavy_wd            |       81.61 | weight_decay(0.0005→0.01)               | sgd          |  0.001   |
| 10_resnet18            |       79.83 | model(resnet34→resnet18)                | sgd          |  0.001   |
| 11_resnet50            |       82.64 | model(resnet34→resnet50)                | sgd          |  0.001   |
| 12_small_batch         |       82.16 | lr(0.001→0.00025)<br>batch_size(128→32) | sgd          |  0.00025 |
| 13_large_batch         |       81.84 | lr(0.001→0.002)<br>batch_size(128→256)  | sgd          |  0.002   |
| 14_no_momentum         |       69.83 | momentum(0.9→0.0)                       | sgd          |  0.001   |

--- 

## 🔍 실험별 상세 분석
### 📍 실험 01_baseline
- **변경 사항:** 🚀 **Base Experiment (Standard)**
- **최종 성능:** Test Accuracy **81.83%** (Best Val: 82.96%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![01_baseline Curve](study_results/01_baseline/resnet34_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 02_high_lr
- **변경 사항:** **lr**: 0.001 → 0.1
- **최종 성능:** Test Accuracy **77.49%** (Best Val: 77.72%)
- **세부 설정:** resnet34 | sgd | LR: 0.1 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![02_high_lr Curve](study_results/02_high_lr/resnet34_f63_sgd_lr0.1_bs128/curves.png)

---
### 📍 실험 03_adam_standard
- **변경 사항:** **optimizer**: sgd → adam, **lr**: 0.001 → 0.0001
- **최종 성능:** Test Accuracy **80.94%** (Best Val: 81.88%)
- **세부 설정:** resnet34 | adam | LR: 0.0001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![03_adam_standard Curve](study_results/03_adam_standard/resnet34_f63_adam_lr0.0001_bs128/curves.png)

---
### 📍 실험 04_freeze_fc
- **변경 사항:** **freeze_level**: 63 → 1
- **최종 성능:** Test Accuracy **61.31%** (Best Val: 62.26%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 1

#### 📈 Learning Curves
![04_freeze_fc Curve](study_results/04_freeze_fc/resnet34_f1_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 05_freeze_partial_deep
- **변경 사항:** **freeze_level**: 63 → 7
- **최종 성능:** Test Accuracy **78.57%** (Best Val: 79.22%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 7

#### 📈 Learning Curves
![05_freeze_partial_deep Curve](study_results/05_freeze_partial_deep/resnet34_f7_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 06_freeze_bottom_up
- **변경 사항:** **freeze_level**: 63 → 33
- **최종 성능:** Test Accuracy **69.31%** (Best Val: 70.00%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 33

#### 📈 Learning Curves
![06_freeze_bottom_up Curve](study_results/06_freeze_bottom_up/resnet34_f33_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 07_high_dropout
- **변경 사항:** **drop_out**: 0.3 → 0.5
- **최종 성능:** Test Accuracy **81.83%** (Best Val: 82.96%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![07_high_dropout Curve](study_results/07_high_dropout/resnet34_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 08_no_smoothing
- **변경 사항:** **label_smoothing**: 0.1 → 0.0
- **최종 성능:** Test Accuracy **81.55%** (Best Val: 82.94%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![08_no_smoothing Curve](study_results/08_no_smoothing/resnet34_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 09_heavy_wd
- **변경 사항:** **weight_decay**: 0.0005 → 0.01
- **최종 성능:** Test Accuracy **81.61%** (Best Val: 82.34%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![09_heavy_wd Curve](study_results/09_heavy_wd/resnet34_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 10_resnet18
- **변경 사항:** **model**: resnet34 → resnet18
- **최종 성능:** Test Accuracy **79.83%** (Best Val: 80.88%)
- **세부 설정:** resnet18 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![10_resnet18 Curve](study_results/10_resnet18/resnet18_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 11_resnet50
- **변경 사항:** **model**: resnet34 → resnet50
- **최종 성능:** Test Accuracy **82.64%** (Best Val: 83.46%)
- **세부 설정:** resnet50 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![11_resnet50 Curve](study_results/11_resnet50/resnet50_f63_sgd_lr0.001_bs128/curves.png)

---
### 📍 실험 12_small_batch
- **변경 사항:** **lr**: 0.001 → 0.00025, **batch_size**: 128 → 32
- **최종 성능:** Test Accuracy **82.16%** (Best Val: 83.42%)
- **세부 설정:** resnet34 | sgd | LR: 0.00025 | BS: 32 | Freeze: 63

#### 📈 Learning Curves
![12_small_batch Curve](study_results/12_small_batch/resnet34_f63_sgd_lr0.00025_bs32/curves.png)

---
### 📍 실험 13_large_batch
- **변경 사항:** **lr**: 0.001 → 0.002, **batch_size**: 128 → 256
- **최종 성능:** Test Accuracy **81.84%** (Best Val: 82.94%)
- **세부 설정:** resnet34 | sgd | LR: 0.002 | BS: 256 | Freeze: 63

#### 📈 Learning Curves
![13_large_batch Curve](study_results/13_large_batch/resnet34_f63_sgd_lr0.002_bs256/curves.png)

---
### 📍 실험 14_no_momentum
- **변경 사항:** **momentum**: 0.9 → 0.0
- **최종 성능:** Test Accuracy **69.83%** (Best Val: 70.42%)
- **세부 설정:** resnet34 | sgd | LR: 0.001 | BS: 128 | Freeze: 63

#### 📈 Learning Curves
![14_no_momentum Curve](study_results/14_no_momentum/resnet34_f63_sgd_lr0.001_bs128/curves.png)

---
