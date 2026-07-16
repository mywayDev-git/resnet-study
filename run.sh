#!/bin/bash

# 파일명 확인 (실제 파일명이 resNet_test.py인지 확인하세요)
PY_FILE="resNet.py"
SAVE_ROOT="study_results"

# 공통 설정
EPOCHS=30
BATCH=128

echo "=========================================================="
echo "   CIFAR-100 Model & Strategy Comparison Study            "
echo "=========================================================="

# ----------------------------------------------------------
# [STEP 1] ResNet34 Baseline (기준점)
# ----------------------------------------------------------
echo "[Step 01] ResNet34 Baseline (Full Tuning)"
python $PY_FILE --model resnet34 --freeze_level 63 --optimizer sgd --lr 0.001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/01_resnet34_base

# ----------------------------------------------------------
# [STEP 2] EfficientNet-B0 vs ResNet34 (Architecture 비교)
# ----------------------------------------------------------
echo "[Step 02] EfficientNet-B0 Baseline (Full Tuning)"
# EfficientNet은 Adam, LR 0.0001이 일반적으로 안정적입니다.
python $PY_FILE --model efficientnet_b0 --freeze_level 63 --optimizer adam --lr 0.0001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/02_efficientnet_base

# ----------------------------------------------------------
# [STEP 3] EfficientNet 동결 전략 (Transfer Learning 비교)
# ----------------------------------------------------------
echo "[Step 03-1] EfficientNet: Head Only (Freeze 1)"
python $PY_FILE --model efficientnet_b0 --freeze_level 1 --optimizer adam --lr 0.0005 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/03_eff_freeze_head

echo "[Step 03-2] EfficientNet: Partial (Head + Late Blocks - Freeze 3)"
# Bit 0(Head) + Bit 1(Late Blocks) = 3
python $PY_FILE --model efficientnet_b0 --freeze_level 3 --optimizer adam --lr 0.0002 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/04_eff_freeze_partial

# ----------------------------------------------------------
# [STEP 4] Optimizer 비교 (EfficientNet 전용)
# ----------------------------------------------------------
echo "[Step 04] EfficientNet: SGD vs Adam Comparison"
python $PY_FILE --model efficientnet_b0 --freeze_level 63 --optimizer sgd --lr 0.01 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/05_eff_sgd_test

# ----------------------------------------------------------
# [STEP 5] 분석 및 리포트 생성
# ----------------------------------------------------------
echo "=========================================================="
echo "   Generating Final Analysis Report...                    "
echo "=========================================================="
python summarize_results.py

echo "모든 실험과 분석이 완료되었습니다. '$SAVE_ROOT' 폴더의 README.md를 확인하세요!"