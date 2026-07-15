#!/bin/bash

PY_FILE="resNet.py"
SAVE_ROOT="study_results"

# 기본 공통 설정
MODEL_DEF="resnet34"
EPOCHS_DEF=100 
BATCH_DEF=256
LR_DEF=0.001

echo "=========================================================="
echo "   CIFAR-100 Full Master Study: 14 Diverse Scenarios      "
echo "=========================================================="

# ----------------------------------------------------------
# 그룹 1: 옵티마이저 & 학습률 최적화 (Optimization)
# ----------------------------------------------------------
echo "[01] Baseline (SGD + LR 0.001)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --optimizer sgd --lr 0.001 --save_dir $SAVE_ROOT/01_baseline

echo "[02] High LR Discovery (SGD + LR 0.1)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --optimizer sgd --lr 0.1 --save_dir $SAVE_ROOT/02_high_lr

echo "[03] Adam Fast Convergence (Adam + LR 0.0001)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --optimizer adam --lr 0.0001 --save_dir $SAVE_ROOT/03_adam_standard

# ----------------------------------------------------------
# 그룹 2: 전이 학습 & 레이어 동결 전략 (Transfer Learning)
# ----------------------------------------------------------
echo "[04] Extreme Freeze (FC only - Level 1)"
python $PY_FILE --model $MODEL_DEF --freeze_level 1 --optimizer sgd --lr 0.001 --save_dir $SAVE_ROOT/04_freeze_fc

echo "[05] Deep Fine-tuning (FC + L4 + L3 - Level 7)"
python $PY_FILE --model $MODEL_DEF --freeze_level 7 --optimizer sgd --lr 0.001 --save_dir $SAVE_ROOT/05_freeze_partial_deep

echo "[06] Bottom-Up (Only Stem + L1 + FC - Level 33)" # 32(Stem)+1(FC)=33
python $PY_FILE --model $MODEL_DEF --freeze_level 33 --optimizer sgd --lr 0.001 --save_dir $SAVE_ROOT/06_freeze_bottom_up

# ----------------------------------------------------------
# 그룹 3: 과적합 방지 및 규제 (Regularization)
# ----------------------------------------------------------
echo "[07] High Dropout (Dropout 0.5)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --drop_out 0.5 --save_dir $SAVE_ROOT/07_high_dropout

echo "[08] No Label Smoothing (Smoothing 0.0)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --label_smoothing 0.0 --save_dir $SAVE_ROOT/08_no_smoothing

echo "[09] Heavy Weight Decay (WD 0.01)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --weight_decay 0.01 --save_dir $SAVE_ROOT/09_heavy_wd

# ----------------------------------------------------------
# 그룹 4: 모델 구조 및 배치 크기 (Architecture & Batch)
# ----------------------------------------------------------
echo "[10] Light Architecture (ResNet18)"
python $PY_FILE --model resnet18 --freeze_level 63 --save_dir $SAVE_ROOT/10_resnet18

echo "[11] Heavy Architecture (ResNet50)"
python $PY_FILE --model resnet50 --freeze_level 63 --save_dir $SAVE_ROOT/11_resnet50

echo "[12] Small Batch Size (Batch 32)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --batch_size 32 --lr 0.00025 --save_dir $SAVE_ROOT/12_small_batch

echo "[13] Large Batch Size (Batch 256)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --batch_size 256 --lr 0.002 --save_dir $SAVE_ROOT/13_large_batch

# ----------------------------------------------------------
# 그룹 5: 모멘텀 전략 (Momentum)
# ----------------------------------------------------------
echo "[14] No Momentum (Momentum 0.0)"
python $PY_FILE --model $MODEL_DEF --freeze_level 63 --optimizer sgd --momentum 0.0 --save_dir $SAVE_ROOT/14_no_momentum

echo "[Final Step] Generating Master Report and Visualizations..."
python summarize_results.py

echo "=========================================================="
echo "   All Experiments & Analysis Completed!                  "
echo "   Check 'master_study_results/README.md' for the report. "
echo "=========================================================="