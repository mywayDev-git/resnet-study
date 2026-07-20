#!/bin/bash

PY_FILE="resNet.py"
SAVE_ROOT="study_results"

# 공통 설정
MODEL="resnet34"
LR=0.001
OPT="sgd"
EPOCHS=25
BATCH=128

echo "=========================================================="
echo "   CIFAR-100 Master Freeze Strategy: 10 Scenarios         "
echo "=========================================================="

# ----------------------------------------------------------
# [그룹 1] 표준 점진적 해제 (Baselines)
# ----------------------------------------------------------
# 01 (000001): 머리만
echo "[01] Head Only"
python $PY_FILE --model $MODEL --freeze_level 1 --save_dir $SAVE_ROOT/01_head_only

# 63 (111111): 전체 다
echo "[02] Full Fine-tuning"
python $PY_FILE --model $MODEL --freeze_level 63 --save_dir $SAVE_ROOT/02_full_tune

# ----------------------------------------------------------
# [그룹 2] 입구와 출구 집중 (The Sandwich Strategy)
# ----------------------------------------------------------
# 33 (100001): Stem + Head (입구와 출구)
echo "[03] Stem + Head"
python $PY_FILE --model $MODEL --freeze_level 33 --save_dir $SAVE_ROOT/03_stem_head

# 35 (100011): Stem + L4 + Head (입구 + 가장 깊은 층 + 출구)
# 기초 특징과 고차원 추상화를 동시에 적응시킴
echo "[04] Stem + L4 + Head"
python $PY_FILE --model $MODEL --freeze_level 35 --save_dir $SAVE_ROOT/04_stem_l4_head

# ----------------------------------------------------------
# [그룹 3] 하위 특징 적응 (Low-Level Adaptation)
# ----------------------------------------------------------
# 49 (110001): Stem + L1 + Head
# 모델의 '눈(Eye)'에 해당하는 앞부분만 수정하여 데이터 도메인에 적응
echo "[05] Low-Level Focus (Stem+L1+Head)"
python $PY_FILE --model $MODEL --freeze_level 49 --save_dir $SAVE_ROOT/05_low_level_focus

# ----------------------------------------------------------
# [그룹 4] 중간층의 반란 (Mid-Level Importance)
# ----------------------------------------------------------
# 13 (001101): L2 + L3 + Head (중간층만 학습)
# L4(추상 개념)는 고정하고 중간 단계의 텍스처/패턴만 학습
echo "[06] Mid-Level Focus (L2+L3+Head)"
python $PY_FILE --model $MODEL --freeze_level 13 --save_dir $SAVE_ROOT/06_mid_level_focus

# ----------------------------------------------------------
# [그룹 5] 징검다리 학습 (Sparse Tuning)
# ----------------------------------------------------------
# 21 (010101): Stem(X), L1(O), L2(X), L3(O), L4(X), Head(O)
# 한 층 건너 하나씩 학습시켜 효율성 테스트
echo "[07] Sparse Tuning (L1+L3+Head)"
python $PY_FILE --model $MODEL --freeze_level 21 --save_dir $SAVE_ROOT/07_sparse_tuning

# ----------------------------------------------------------
# [그룹 6] 특정 블록 고립 학습 (Isolated Block Study)
# ----------------------------------------------------------
# 5 (000101): L3 + Head (다른 모든 몸통 고정)
echo "[08] L3 Isolation + Head"
python $PY_FILE --model $MODEL --freeze_level 5 --save_dir $SAVE_ROOT/08_l3_isolation

# 9 (001001): L2 + Head
echo "[09] L2 Isolation + Head"
python $PY_FILE --model $MODEL --freeze_level 9 --save_dir $SAVE_ROOT/09_l2_isolation

# ----------------------------------------------------------
# [그룹 7] 몸통만 학습 (The Headless Experiment)
# ----------------------------------------------------------
# 62 (111110): Head 제외 모든 몸통 학습
# 사전 학습된 Head가 이미 충분한지, 아니면 Head는 무조건 학습해야 하는지 확인
echo "[10] No Head Training (Body Only)"
python $PY_FILE --model $MODEL --freeze_level 62 --save_dir $SAVE_ROOT/10_no_head_tune

# ----------------------------------------------------------
# 분석 리포트 자동 생성
# ----------------------------------------------------------
python summarize_results.py

echo "모든 시나리오 학습 완료!"