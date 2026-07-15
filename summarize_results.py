import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 서버/도커 환경 대응
plt.switch_backend('agg')

# ==========================================
# [Strategy 2] 주요 파라미터 화이트리스트 정의
# ==========================================
MAJOR_KEYS = [
    'model',          # 모델 아키텍처
    'optimizer',      # 옵티마이저 종류
    'lr',             # 학습률
    'freeze_level',   # 레이어 동결 정도
    'batch_size',     # 배치 크기
    'drop_out',       # 드롭아웃 비율
    'weight_decay',   # 가중치 감쇄
    'label_smoothing',# 라벨 스무딩
    'fc_layers',      # FC 레이어 깊이
    'fc_dim',         # FC 레이어 너비
    'momentum'        # 모멘텀
]

def get_diff(baseline, current, is_table=False):
    """화이트리스트에 있는 키값들만 비교하여 유의미한 차이점 추출"""
    diffs = []
    
    # 화이트리스트(MAJOR_KEYS)에 정의된 항목만 비교 순회
    for k in MAJOR_KEYS:
        if k in baseline and k in current:
            # 값 자체가 다를 경우에만 변경점으로 간주
            if str(baseline[k]) != str(current[k]):
                if is_table:
                    # 표용 간결한 형식
                    diffs.append(f"{k}({baseline[k]}→{current[k]})")
                else:
                    # 상세 섹션용 강조 형식
                    diffs.append(f"**{k}**: {baseline[k]} → {current[k]}")
    
    if not diffs:
        return "-" if is_table else "🚀 **Base Experiment (Standard)**"
    
    return "<br>".join(diffs) if is_table else ", ".join(diffs)

def generate_summary(root_dir_name="study_results"):
    base_path = os.path.abspath(os.getcwd())
    target_path = os.path.join(base_path, root_dir_name)
    
    if not os.path.exists(target_path):
        print(f"❌ '{root_dir_name}' 폴더를 찾을 수 없습니다.")
        return

    all_data = []
    
    # 데이터 수집 (깊은 폴더 구조 대응)
    print("🔍 유의미한 실험 데이터를 필터링하며 스캔 중...")
    for root, dirs, files in os.walk(target_path):
        if "summary.json" in files:
            json_path = os.path.join(root, "summary.json")
            
            rel_to_target = os.path.relpath(root, target_path)
            parts = rel_to_target.split(os.sep)
            exp_id = parts[0]
            
            curve_img = None
            for f in files:
                if f in ["curves.png", "learning_curves.png", "accuracy_comparison.png"]:
                    curve_img = os.path.join(rel_to_target, f)
                    break

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data['exp_id'] = exp_id
                    data['curve_img'] = curve_img
                    all_data.append(data)
            except Exception as e:
                print(f"❌ 로드 실패: {json_path} ({e})")

    if not all_data:
        print("❗ 분석할 데이터를 찾지 못했습니다.")
        return

    # 실험 순서 정렬
    all_data.sort(key=lambda x: x['exp_id'])
    baseline_params = all_data[0]

    # 변경점 계산 (화이트리스트 기반)
    for i, exp in enumerate(all_data):
        exp['changes_table'] = get_diff(baseline_params, exp, is_table=True)
        exp['changes_detail'] = get_diff(baseline_params, exp, is_table=False)

    # 데이터프레임 및 CSV
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(target_path, "final_analysis.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # README.md 작성
    readme_path = os.path.join(base_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 CIFAR-100 실험 히스토리 및 분석 리포트\n\n")
        f.write(f"> **총 실험 개수:** {len(all_data)}개 | **업데이트:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 📊 전체 요약 (주요 변경점 위주)\n")
        # 표 상단 구성
        summary_table = df[['exp_id', 'test_acc', 'changes_table', 'optimizer', 'lr']].copy()
        summary_table.columns = ['실험 ID', '정확도(%)', '주요 파라미터 변화', '옵티마이저', '학습률']
        f.write(summary_table.to_markdown(index=False) + "\n\n")
        
        f.write("--- \n\n")
        f.write("## 🔍 실험별 상세 분석\n")

        for i, exp in enumerate(all_data):
            f.write(f"### 📍 실험 {exp['exp_id']}\n")
            f.write(f"- **변경 사항:** {exp['changes_detail']}\n")
            f.write(f"- **최종 성능:** Test Accuracy **{exp['test_acc']:.2f}%** (Best Val: {exp['best_val_acc']:.2f}%)\n")
            f.write(f"- **세부 설정:** {exp['model']} | {exp['optimizer']} | LR: {exp['lr']} | BS: {exp['batch_size']} | Freeze: {exp['freeze_level']}\n")
            
            if exp['curve_img']:
                f.write(f"\n#### 📈 Learning Curves\n")
                f.write(f"![{exp['exp_id']} Curve]({root_dir_name}/{exp['curve_img']})\n\n")
            
            f.write("---\n")

    print(f"\n✅ 화이트리스트 기반 분석 완료!")
    print(f"📑 리포트 확인: {os.path.abspath(readme_path)}")

if __name__ == "__main__":
    generate_summary("study_results")