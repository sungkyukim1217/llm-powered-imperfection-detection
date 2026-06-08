import pandas as pd
import numpy as np
import re

def extract_label_value(label_str):
    if pd.isna(label_str):
        return np.nan
    match = re.search(r'\((.*?)\)', str(label_str))
    return match.group(1) if match else label_str

def run_evaluation(df):
    print("\n" + "="*50)
    print("      PROCESS MINING LABEL REFINEMENT EVALUATION")
    print("="*50)
    
    eval_df = df.copy()
    eval_df['true_label'] = eval_df['label'].apply(extract_label_value)
    
    # [1] 이번 턴(Current Turn)의 성능 계산 
    # 오직 이번 턴에 LLM이 새로 뱉은 'restored_activity'만 평가합니다.
    turn_tp = len(eval_df[(eval_df['true_label'].notna()) & 
                         (eval_df['true_label'] == eval_df['restored_activity'])])
    turn_predicted = eval_df['restored_activity'].notna().sum()
    turn_actual = eval_df['true_label'].notna().sum() # 전체 정답 수 (Recall 기준용)
    
    turn_precision = turn_tp / turn_predicted if turn_predicted > 0 else 0
    # 이번 턴의 Recall은 전체 정답 중 이번에 LLM이 새로 맞춘 비율입니다.
    turn_recall = turn_tp / turn_actual if turn_actual > 0 else 0
    turn_f1 = 2 * (turn_precision * turn_recall) / (turn_precision + turn_recall) if (turn_precision + turn_recall) > 0 else 0

    # --- 데이터 업데이트 (activity 열에 이번 결과 반영) ---
    mask = eval_df['restored_activity'].notna()
    eval_df.loc[mask, 'activity'] = eval_df.loc[mask, 'restored_activity']

    # [2] 누적(Accumulated) 성능 계산
    # 업데이트가 완료된 'activity' 열 전체를 정답 'true_label'과 비교합니다.
    cum_tp = len(eval_df[(eval_df['true_label'].notna()) & 
                        (eval_df['true_label'] == eval_df['activity'])])
    
    cum_precision = cum_tp / turn_actual if turn_actual > 0 else 0 # 정밀도 (누적)
    cum_recall = cum_tp / turn_actual if turn_actual > 0 else 0    # 재현율 (누적)
    cum_f1 = 2 * (cum_precision * cum_recall) / (cum_precision + cum_recall) if (cum_precision + cum_recall) > 0 else 0

    # [3] 결과 출력
    print(f"  [TURN STATS - 이번 턴 성적]")
    print(f"  - Turn Predictions: {turn_predicted}")
    print(f"  - Turn Correct (TP): {turn_tp}")
    print(f"  - Turn Precision: {turn_precision:.4f}")
    
    print("-" * 30)
    print(f"  [ACCUMULATED STATS - 누적 합계]")
    print(f"  - Total Actual Homonyms: {turn_actual}")
    print(f"  - Cumulative Correct: {cum_tp}")
    print(f"  - Cumulative F1 (Progress): {cum_f1:.4f}")
    print("="*50)

    # 딕셔너리에 두 가지 지표를 모두 담아서 반환
    results = {
        'turn_metrics': {
            'precision': turn_precision,
            'recall': turn_recall,
            'f1': turn_f1,
            'tp': turn_tp,
            'predicted': turn_predicted
        },
        'cumulative_metrics': {
            'f1': cum_f1,
            'tp': cum_tp,
            'total_actual': turn_actual
        }
    }

    return results, eval_df