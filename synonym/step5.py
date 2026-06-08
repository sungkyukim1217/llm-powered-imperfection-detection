from pm4py.util import constants
import pandas as pd
import json
import pm4py
from util import llm_gen

constants.SHOW_PROGRESS_BAR = False
def analyze_variants(df, name="Dataset", variant_coverage = 0.8, variant_threshold = 1.0):
    """
    Analyzes and summarizes the most frequent process variants within a given event log.
    
    This function profiles the 'behavioral backbone' of a dataset by extracting unique execution 
    paths (variants) and ranking them by frequency. It helps the LLM understand the standard 
    operating procedures of a process by focusing on a specific coverage percentage or 
    frequency threshold, filtering out rare noise/outliers.

    Args:
        df: The DataFrame to analyze.
        name: A descriptive name for the dataset (used in output logs).
        variant_coverage: Cumulative frequency percentage to include (e.g., top 80% of cases).
        variant_threshold: Minimum frequency percentage for a single variant to be included.

    Returns:
        dict: A summary containing total cases, coverage limits, and a list of ranked variants with their paths.
    """
    
    if df.empty: 
        return {"dataset_name": name, "status": "Empty Dataset", "total_cases": 0, "variants": []}
    log = df[['case_id', 'timestamp', 'activity']].copy()
    log.rename(columns={'case_id': 'case:concept:name', 'timestamp': 'time:timestamp', 'activity': 'concept:name'}, inplace=True)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    total_cases = log['case:concept:name'].nunique()
    target_val, cumulative, result_v = total_cases * variant_coverage, 0, []
    variants = pm4py.get_variants(log)
    sorted_v = sorted(variants.items(), key=lambda x: x[1] if isinstance(x[1], int) else len(x[1]), reverse=True)
    for rank, (variant, val) in enumerate(sorted_v, 1):
        count = val if isinstance(val, int) else len(val); percentage = (count / total_cases) * 100
        if percentage < variant_threshold: 
            break
        cumulative += count; result_v.append({"rank": rank, "frequency": count, "percentage": f"{percentage:.1f}%", "path": " -> ".join(variant)})
        if cumulative >= target_val: 
            break
                
    return {"dataset_name": name,
            "total_cases": total_cases,
            "analysis_limit": f"{int(variant_coverage*100)}% Coverage or {variant_threshold}% Min_Freq",
            "variants": result_v,
            "total_unique_variants": len(sorted_v)}
    
def split_cases_analysis(df, target_act, original_acts, is_step4_1):
    """
    Partitions the event log into three distinct subsets based on the presence of target and legacy labels.
    
    This function prepares the data for 'Path Substitution Simulation' by categorizing cases into:
    1. Homonymous_Labels_Cases: Any case containing the target label (including mixed cases).
    2. Original_Labels_Cases: Legacy cases containing ONLY original labels.
    

    Args:
        df: The combined event log DataFrame.
        target_act: The simplified (homonymous) activity name.
        original_acts: A list of detailed legacy activity names.

    Returns:
        tuple: (df_target_only, df_origin_only) partitioned DataFrames.
    """
    
    case_contents = df.groupby('case_id')['activity'].apply(set).to_dict()
    target_total_case_ids, origin_only_case_ids = [], []   
    original_acts_set = set(original_acts)
    
    for case_id, acts in case_contents.items():
        has_target = target_act in acts
        has_origin = not acts.isdisjoint(original_acts_set)
        if has_target:
            target_total_case_ids.append(case_id)
       # elif has_origin:
        else:
            origin_only_case_ids.append(case_id)
    def rename_activity(act):
        if act in original_acts_set:
            return f"{target_act} (originally: {act})"
        return act
    df_target_total = df[df['case_id'].isin(target_total_case_ids)].copy()
    df_origin_only = df[df['case_id'].isin(origin_only_case_ids)].copy()
    if (not df_origin_only.empty) and (is_step4_1==True):
        df_origin_only['activity'] = df_origin_only['activity'].replace(original_acts, target_act)
        
    return df_target_total, df_origin_only

def run_step5(llm, model_, llm_repetition, filtered_candidates, df_new, sys_prompt, user_prompt_tmpl):

    print(f"\n>>> Running Step 4 with {llm_repetition} repetitions...")
    
    synonym_predict_5 = {}

    for target_name, candidates in filtered_candidates.items():
        for combo in candidates:
            combo_tuple = tuple(combo)
            
            # 1. 데이터셋 분석 (반복문 밖에서 1회 수행)
            df_t_2, df_o_2 = split_cases_analysis(df_new, target_name, combo, False)
            origin_var_2 = json.dumps(analyze_variants(df_o_2, name="Original_Labels_Cases"), indent=4, ensure_ascii=False)
            target_var_2 = json.dumps(analyze_variants(df_t_2, name="Homonymous_Labels_Cases"), indent=4, ensure_ascii=False)
            user_prompt_2 = user_prompt_tmpl_2.format(
                HOMONYM_STEP4_2_INPUT1 = target_name,
                HOMONYM_STEP4_2_INPUT2 = combo,
                HOMONYM_STEP4_2_INPUT3 = origin_var_2,
                HOMONYM_STEP4_2_INPUT4 = target_var_2
            )
            prompt_2 = [
                {"role": "system", "content": sys_prompt_2},
                {"role": "user", "content": user_prompt_2}
            ]
            true_count_2 = 0
            for i in range(llm_repetition):
                res2 = llm_gen(model_version=model_, model_instance=llm, prompt=prompt_2)
                if res2.get('is_homonym') is True:
                    true_count_2 += 1

            threshold = llm_repetition / 2
            confirmed_2 = true_count_2 >= threshold
            print(f"  - Validating: {target_name} -> {combo}")
            if confirmed_2:
                if target_name not in homonym_predict_4:
                    homonym_predict_4[target_name] = []
                homonym_predict_4[target_name].append(combo)
                
                print(f"    => [CONFIRMED] P2 validation passed: {true_count_2}/{llm_repetition}")
            else:
                print(f"    => [REJECTED] P2 validation failed: {true_count_2}/{llm_repetition}")
                
    target_keys = list(homonym_predict_4.keys())
    print(f"Mapping Preview (Total: {len(target_keys)} groups)")

    for target in target_keys[:3]: 
        candidates = homonym_predict_4[target]
        print(f"  - Target: [{target}]")
        print(f"    └─ Validated: {candidates[0]} ({len(candidates)} total groups)")
    if len(target_keys) > 3:
        print(f"  ... and {len(target_keys) - 3} more targets.")
    return homonym_predict_4