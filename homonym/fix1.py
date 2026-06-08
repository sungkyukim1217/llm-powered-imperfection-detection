from pm4py.util import constants
import pandas as pd
import json
from util import llm_gen
import pm4py
from collections import Counter
constants.SHOW_PROGRESS_BAR = False

def analyze_variants(df, name="Dataset", variant_coverage = 0.8, variant_threshold = 1.0):
    """
    Same as step4's
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
    Same as step4's
    Partitions the event log into two distinct subsets based on the presence of target and legacy labels.
    
    This function prepares the data for 'Path Substitution Simulation' by categorizing cases into:
    1. Homonymous_Labels_Cases: Cases containing only the simplified target label.
    2. Original_Labels_Cases: Legacy cases containing only the specific original (detailed) labels.

    Args:
        df: The combined event log DataFrame.
        target_act: The simplified (homonymous) activity name.
        original_acts: A list of detailed legacy activity names.

    Returns:
        tuple: (df_target_only, df_mixed, df_origin_only) partitioned DataFrames.
    """

    
    case_contents = df.groupby('case_id')['activity'].apply(set).to_dict()
    target_total_case_ids, origin_only_case_ids = [], []   
    original_acts_set = set(original_acts)
    
    for case_id, acts in case_contents.items():
        has_target = target_act in acts
        has_origin = not acts.isdisjoint(original_acts_set)
        if has_target:
            target_total_case_ids.append(case_id)
        elif has_origin:
            origin_only_case_ids.append(case_id)
            
    df_target_total = df[df['case_id'].isin(target_total_case_ids)].copy()
    df_origin_only = df[df['case_id'].isin(origin_only_case_ids)].copy()
    
    return df_target_total, df_origin_only


def run_fix1(llm, model_, llm_repetition, finalized_mapping, df, sys_prompt, user_prompt_tmpl):
    fix1_results = {}
    for target_name, selected_candidates in finalized_mapping.items():
        print(f"\n  - Restoring Traces for: {target_name} ({llm_repetition} Iterations)")
        df_t, df_o = split_cases_analysis(df, target_name, selected_candidates)
        target_var_json = analyze_variants(df_t, name="Homonymous_Labels_Cases")
        input_ranks = {int(v['rank']) for v in target_var_json.get('variants', []) if 'rank' in v}
        if not input_ranks:
            print(f"    [Error] No ranks found in Dataset 2 for {target_name}.")
            continue
        origin_var_str = json.dumps(analyze_variants(df_o, name="Original_Labels_Cases"), indent=4, ensure_ascii=False)
        target_var_str = json.dumps(target_var_json, indent=4, ensure_ascii=False)
        user_prompt = user_prompt_tmpl.format(
            HOMONYM_FIX1_INPUT1 = target_name,
            HOMONYM_FIX1_INPUT2 = selected_candidates,
            HOMONYM_FIX1_INPUT3 = origin_var_str,
            HOMONYM_FIX1_INPUT4 = target_var_str
        )
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        collected_rank_paths = {rank: [] for rank in input_ranks}
        for i in range(llm_repetition):
            success = False
            fail_count = 0
            while not success:
                response = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
                if not response or not isinstance(response, dict):
                    fail_count += 1
                    print(f"    [Retry] Invalid JSON (Fail: {fail_count})")
                    continue
                try:
                    response_ranks = {int(k) for k in response.keys() if str(k).isdigit()}
                except ValueError:
                    fail_count += 1
                    continue
                if input_ranks == response_ranks:
                    for r_str, path in response.items():
                        collected_rank_paths[int(r_str)].append(path)
                    success = True
                else:
                    fail_count += 1
                    print(f"    [Retry] Rank Mismatch (Fail: {fail_count})")
        final_restored_map = {}
        for rank in sorted(input_ranks):
            paths = collected_rank_paths[rank]
            if paths:
                best_path, _ = Counter(paths).most_common(1)[0]
                final_restored_map[str(rank)] = best_path
        fix1_results[target_name] = final_restored_map
        print(f"    => [DONE] {target_name}: Restoration complete with voting.")
    return fix1_results