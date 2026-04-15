import re
import pandas as pd

def run_evaluation(df_eval, df_original):
    """
    Evaluates the correction by comparing the fixed dataframe (df_eval) 
    with the original raw dataframe (df_original).
    """
    print("\n>>> Starting Evaluation...")

    def extract_ground_truth(label):
        if pd.isna(label):
            return None
        match = re.search(r'\((.*?)\)', str(label))
        return match.group(1) if match else None

    # Prepare evaluation dataframe
    df_eval = df_eval.copy()
    df_eval['ground_truth'] = df_eval['label'].apply(extract_ground_truth)
    
    # Crucial: Use original activity from df_original to detect if it was changed
    # Assuming the index is preserved between df_original and df_eval
    original_activities = df_original['activity']
    
    results = []
    for idx, row in df_eval.iterrows():
        # Get the original activity name using the index
        orig_act = original_activities.loc[idx]
        curr_act = row['activity'] # This is the "Predicted" activity now
        
        is_distorted = pd.notna(row['label'])
        is_corrected = orig_act != curr_act # Compare with original
        
        if is_distorted:
            if is_corrected and curr_act == row['ground_truth']:
                status = "Correct (TP)"
            elif is_corrected and curr_act != row['ground_truth']:
                status = "Wrong Target (FP/FN)"
            else:
                status = "Missed (FN)"
        else:
            if is_corrected:
                status = "Over-correction (FP)"
            else:
                status = "Correct (TN)"
        results.append(status)

    df_eval['eval_status'] = results
    summary = df_eval['eval_status'].value_counts()

    # Calculate metrics
    tp = summary.get("Correct (TP)", 0)
    fn = summary.get("Missed (FN)", 0)
    fp = summary.get("Over-correction (FP)", 0) + summary.get("Wrong Target (FP/FN)", 0)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print("-" * 30)
    print("EVALUATION SUMMARY")
    print("-" * 30)
    print(summary.to_string())
    print("-" * 30)
    print(f"Recall:    {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print("-" * 30)

    # Show samples for debugging
    errors = df_eval[df_eval['eval_status'].isin(["Missed (FN)", "Wrong Target (FP/FN)", "Over-correction (FP)"])]
    if not errors.empty:
        print("\n>>> Error Samples (Top 5):")
        # We use original_activities.loc to show what it was before
        error_df = errors.copy()
        error_df['original_activity'] = original_activities.loc[errors.index]
        print(error_df[['original_activity', 'activity', 'ground_truth', 'eval_status']].head(5))
        
    return summary