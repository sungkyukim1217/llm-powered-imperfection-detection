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

def run_step5(llm, model_, llm_repetition, filtered_homonyms, df, sys_prompt, user_prompt_tmpl):
    """
    Selects the optimal semantic decomposition for a homonymous label through comparative simulation.
    
    This function evaluates multiple 'Candidate Options' for a single homonymous activity and 
    picks the one that best explains the legacy process structure. It ensures that the 
    disaggregation of a generic label (e.g., 'Task') into specific ones (e.g., 'Task_A', 'Task_B') 
    is both minimalist and structurally accurate.

    The selection logic follows a strict "Comparative Reconstruction" protocol:
    1. Multi-Option Simulation: For each candidate list, the LLM performs a mental restoration 
       of the target label within the process variants.
    2. Parsimony & Evidence Check: It prioritizes the most precise list while rejecting 
       'ghost candidates' (labels in the option list that are never actually used in the 
       restoration context).
    3. Structural Superiority: It favors the candidate set whose simulated paths show the 
       highest 'Backbone Alignment' (matching high-frequency variants) with the legacy data.
    4. Consensus Voting: If multiple options exist, the LLM votes 'llm_repetition' times. 
       The most frequently selected combination is designated as the final semantic pattern.

    Args:
        llm: The LLM interface for comparative pattern analysis and decision-making.
        model_: String identifier for the model architecture version.
        llm_repetition: Number of iterations for voting to ensure statistical consensus.
        filtered_homonyms: Validated homonym groups from Step 4.
        df: The event log DataFrame used for variant profiling and splitting.
        sys_prompt: Strategic instructions defining selection priority (Parsimony vs. Backbone).
        user_prompt_tmpl: A template for presenting candidate options and baseline dataset variants.

    Returns:
        dict: A finalized mapping dictionary where each homonymous label is assigned 
              to exactly one optimized list of original activities.
    """
    print("\n>>> Running Step 5: Final Semantic Pattern Selection")
    
    final_prediction_results = {}

    for target_name, candidates in filtered_homonyms.items():
        if len(candidates) == 1:
            print(f"  - Finalizing Patterns for: {target_name} (Auto-selected: Single candidate)")
            final_prediction_results[target_name] = candidates[0]
            continue
        print(f"  - Finalizing Patterns for: {target_name} (LLM Voting required)")
        merged_list = list(set(item for sublist in candidates for item in sublist))
        df_t, df_m, df_o = split_cases_analysis(df, target_name, merged_list)
        df_t = pd.concat([df_t, df_m], ignore_index=True)
        origin_var = json.dumps(analyze_variants(df_o, name="Original_Labels_Cases"), indent=4, ensure_ascii=False)
        target_var = json.dumps(analyze_variants(df_t, name="Homonymous_Labels_Cases"), indent=4, ensure_ascii=False)
        candidates = json.dumps(candidates, indent=4, ensure_ascii=False)
        user_prompt = user_prompt_tmpl.format(
            HOMONYM_STEP5_INPUT1 = target_name,
            HOMONYM_STEP5_INPUT2 = candidates,
            HOMONYM_STEP5_INPUT3 = origin_var,
            HOMONYM_STEP5_INPUT4 = target_var)
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        votes = []

        for i in range(llm_repetition):
            response = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
            selected_combo = tuple(response.get('candidate_originals', []))
            if selected_combo:
                votes.append(selected_combo)
        if votes:
            most_common_combo, count = Counter(votes).most_common(1)[0]
            final_prediction_results[target_name] = list(most_common_combo)            
            print(f"    => [SELECTED] {list(most_common_combo)} ({count}/{len(votes)} votes)")

    return final_prediction_results

