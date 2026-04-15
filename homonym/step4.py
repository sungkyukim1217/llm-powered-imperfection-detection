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
    
def split_cases_analysis(df, target_act, original_acts):
    """
    Partitions the event log into three distinct subsets based on the presence of target and legacy labels.
    
    This function prepares the data for 'Path Substitution Simulation' by categorizing cases into:
    1. Homonymous_Labels_Cases: Cases containing only the simplified target label.
    2. Co-occurring_Labels_Cases: Transitional cases containing both target and legacy labels.
    3. Original_Labels_Cases: Legacy cases containing only the specific original (detailed) labels.

    For comparative consistency, it automatically replaces legacy labels with the target label 
    in the 'Mixed' and 'Original' subsets to allow for structural alignment tests in Step 4.

    Args:
        df: The combined event log DataFrame.
        target_act: The simplified (homonymous) activity name.
        original_acts: A list of detailed legacy activity names.

    Returns:
        tuple: (df_target_only, df_mixed, df_origin_only) partitioned DataFrames.
    """
    
    case_contents = df.groupby('case_id')['activity'].apply(set).to_dict()
    target_only_case_ids, mixed_case_ids, origin_only_case_ids = [], [], []   
    original_acts_set = set(original_acts)
    for case_id, acts in case_contents.items():
        has_target = target_act in acts
        has_origin = not acts.isdisjoint(original_acts_set)
        if has_target and not has_origin:
            target_only_case_ids.append(case_id)
        elif has_target and has_origin:
            mixed_case_ids.append(case_id)
        elif not has_target and has_origin:
            origin_only_case_ids.append(case_id)
    df_target_only = df[df['case_id'].isin(target_only_case_ids)].copy()
    df_mixed = df[df['case_id'].isin(mixed_case_ids)].copy()
    df_origin_only = df[df['case_id'].isin(origin_only_case_ids)].copy()
    if not df_mixed.empty:
        df_mixed['activity'] = df_mixed['activity'].replace(original_acts, target_act)
    if not df_origin_only.empty:
        df_origin_only['activity'] = df_origin_only['activity'].replace(original_acts, target_act)
    return df_target_only, df_mixed, df_origin_only

def run_step4(llm, model_, llm_repetition, filtered_candidates, df_new, sys_prompt_1, user_prompt_tmpl_1, sys_prompt_2, user_prompt_tmpl_2):
    """
    Performs a final, high-fidelity validation of homonyms using dual-perspective path simulations.
    
    This function executes the most rigorous test in the pipeline by forcing the LLM to 
    simulate process executions from two different directions:
    
    Perspective 1 (Substitution): "If we hide the detailed labels, does the process look 
    identical to the simplified one?" (Validates if the target is a proper surrogate).
    Perspective 2 (Reconstruction): "Can we logically break down every instance of the 
    simplified label back into its detailed original parts?" (Validates logical decomposition).

    Validation Logic:
    - Path Profiling: Converts split datasets into variant summaries for the LLM to 'read' the process.
    - Dual Prompting: Uses two distinct sets of System/User prompts to prevent bias and ensure 
      consistency from both 'bottom-up' and 'top-down' viewpoints.
    - Statistical Consensus: Employs a voting threshold for each perspective. If either 
      simulation perspective confirms the homonym relationship with high confidence (>= 50%), 
      the candidate is officially validated.

    Args:
        llm: The LLM interface for structural and behavioral simulation.
        model_: String identifier for the model version.
        llm_repetition: Number of iterations per perspective to ensure result stability.
        filtered_candidates: Semantic-validated homonym groups from Step 3.
        df_new: The event log DataFrame.
        sys_prompt_1/2: Strategic instructions for Substitution and Reconstruction simulations.
        user_prompt_tmpl_1/2: Templates providing the profiled variant data to the LLM.

    Returns:
        dict: The final, triple-validated dictionary of homonymous mappings.
    """
    print(f"\n>>> Running Step 4 with {llm_repetition} repetitions...")
    
    homonym_predict_4 = {}

    for target_name, candidates in filtered_candidates.items():
        for combo in candidates:
            combo_tuple = tuple(combo)
            
            # 1. 데이터셋 분석 (반복문 밖에서 1회 수행)
            df_t, df_m, df_o = split_cases_analysis(df_new, target_name, combo)
            origin_var = json.dumps(analyze_variants(df_o, name="Original_Labels_Cases"), indent=4, ensure_ascii=False)
            mixed_var = json.dumps(analyze_variants(df_m, name="Co-occurring_Labels_Cases"), indent=4, ensure_ascii=False)
            target_var = json.dumps(analyze_variants(df_t, name="Homonymous_Labels_Cases"), indent=4, ensure_ascii=False)
            user_prompt_1 = user_prompt_tmpl_1.format(
                HOMONYM_STEP4_1_INPUT1 = target_name,
                HOMONYM_STEP4_1_INPUT2 = combo,
                HOMONYM_STEP4_1_INPUT3 = origin_var,
                HOMONYM_STEP4_1_INPUT4 = mixed_var,
                HOMONYM_STEP4_1_INPUT5 = target_var
            )
            prompt_1 = [
                {"role": "system", "content": sys_prompt_1},
                {"role": "user", "content": user_prompt_1}
            ]
            user_prompt_2 = user_prompt_tmpl_2.format(
                HOMONYM_STEP4_2_INPUT1 = target_name,
                HOMONYM_STEP4_2_INPUT2 = combo,
                HOMONYM_STEP4_2_INPUT3 = origin_var,
                HOMONYM_STEP4_2_INPUT4 = mixed_var,
                HOMONYM_STEP4_2_INPUT5 = target_var
            )
            prompt_2 = [
                {"role": "system", "content": sys_prompt_2},
                {"role": "user", "content": user_prompt_2}
            ]
            true_count_1 = 0
            true_count_2 = 0
            for i in range(llm_repetition):
                res1 = llm_gen(model_version=model_, model_instance=llm, prompt=prompt_1)
                if res1.get('is_homonym') is True:
                    true_count_1 += 1
                res2 = llm_gen(model_version=model_, model_instance=llm, prompt=prompt_2)
                if res2.get('is_homonym') is True:
                    true_count_2 += 1

            threshold = llm_repetition / 2
            confirmed_1 = true_count_1 >= threshold
            confirmed_2 = true_count_2 >= threshold

            if confirmed_1 or confirmed_2:
                if target_name not in homonym_predict_4:
                    homonym_predict_4[target_name] = []
                homonym_predict_4[target_name].append(combo)
                
                status = "BOTH" if (confirmed_1 and confirmed_2) else ("P1" if confirmed_1 else "P2")
                print(f"  - Validating: {target_name} -> {combo}")
                print(f"    => [CONFIRMED by {status}] P1: {true_count_1}/{llm_repetition}, P2: {true_count_2}/{llm_repetition}")


    return homonym_predict_4