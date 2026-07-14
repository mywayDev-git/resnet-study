#!/bin/bash

# 파이썬 파일 이름 (본인의 파일명과 일치하는지 확인하세요)
PY_FILE="resNet_test.py"
SAVE_ROOT="study_results"

# 공통 설정 (변하지 않는 값)
MODEL="resnet34"
EPOCHS=30
BATCH=256

echo "=========================================================="
echo "   CIFAR-100 Deep Learning Study: Step-by-Step Tests      "
echo "=========================================================="

# ----------------------------------------------------------
# [STEP 1] 베이스라인 (Baseline)
# 모든 실험의 기준점입니다. (Full Tuning, SGD, LR 0.001)
# ----------------------------------------------------------
echo "[Step 1] Baseline: Standard Full Tuning"
python $PY_FILE --model $MODEL --freeze_level 63 --optimizer sgd --lr 0.001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/01_baseline


# ----------------------------------------------------------
# [STEP 2] 학습률(Learning Rate) 비교
# 베이스라인에서 '학습률'만 바꿔봅니다.
# ----------------------------------------------------------
echo "[Step 2-1] LR Study: High Learning Rate (0.01)"
python $PY_FILE --model $MODEL --freeze_level 63 --optimizer sgd --lr 0.01 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/02_high_lr

echo "[Step 2-2] LR Study: Low Learning Rate (0.0001)"
python $PY_FILE --model $MODEL --freeze_level 63 --optimizer sgd --lr 0.0001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/03_low_lr


# ----------------------------------------------------------
# [STEP 3] 전이 학습 전략(Freeze Level) 비교
# 베이스라인에서 '동결 레이어'만 바꿔봅니다. (LR은 0.001 유지)
# ----------------------------------------------------------
echo "[Step 3-1] Freeze Study: FC Only (Freeze Level 1)"
python $PY_FILE --model $MODEL --freeze_level 1 --optimizer sgd --lr 0.001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/04_freeze_fc_only

echo "[Step 3-2] Freeze Study: Partial (Freeze Level 7 - FC+L4+L3)"
python $PY_FILE --model $MODEL --freeze_level 7 --optimizer sgd --lr 0.001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/05_freeze_partial


# ----------------------------------------------------------
# [STEP 4] 옵티마이저(Optimizer) 비교
# 베이스라인에서 '옵티마이저'만 Adam으로 바꿔봅니다.
# (Adam은 보통 SGD보다 낮은 LR에서 잘 작동하므로 0.0001 사용)
# ----------------------------------------------------------
echo "[Step 4] Optimizer Study: Adam vs SGD"
python $PY_FILE --model $MODEL --freeze_level 63 --optimizer adam --lr 0.0001 \
                --epochs $EPOCHS --batch_size $BATCH --save_dir $SAVE_ROOT/06_optimizer_adam


echo "=========================================================="
echo "   All Study Experiments Completed!                       "
echo "   Check the '$SAVE_ROOT' directory to compare results.   "
echo "=========================================================="