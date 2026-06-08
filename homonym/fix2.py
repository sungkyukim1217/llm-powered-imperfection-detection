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


def create_batched_jsons2(
    df: pd.DataFrame,
    case_id: str = 'case_id',
    activity: str = 'activity',
    timestamp: str = 'timestamp',
    event_id: str = 'event_id',
    label: str = 'label',
    chunk_cases: int | None = 1,
    use_cols: list[str] | None = None
) -> list[list[dict]]:
    if use_cols is None:
        use_cols = [event_id, activity]#list(df.columns)
    input_cols = [c for c in use_cols if c not in {label}]
    cases = []
    for cid, g in df.groupby(case_id, sort=False):
        events = []
        for _, row in g.iterrows():
            e = {k: str(row[k]) for k in input_cols}
            events.append(e)
        cases.append(events)
    batched_cases = []
    if chunk_cases is None or chunk_cases <= 0:
        chunk_cases = 1
    for i in range(0, len(cases), chunk_cases):
        batch_cases = cases[i:i + chunk_cases]
        batch_events = [e for case in batch_cases for e in case]
        batched_cases.append(batch_events)
    return batched_cases


def create_batched_jsons(
    df: pd.DataFrame,
    target_name: str,        
    case_id: str = 'case_id',
    activity: str = 'activity',
    timestamp: str = 'timestamp',
    event_id: str = 'event_id',
    label: str = 'label',
    chunk_cases: int | None = 1,
    use_cols: list[str] | None = None
) -> list[list[dict]]:
    if use_cols is None:
        use_cols = [event_id, activity]#list(df.columns)
    input_cols = [c for c in use_cols if c not in {label}]
    cases = []
    for cid, g in df.groupby(case_id, sort=False):
        events = []
        for _, row in g.iterrows():
            event_dict = {}
            for k in input_cols:
                val = str(row[k])
                if k == 'activity' and val == target_name:
                    event_dict['homonymous_activity'] = val
                else:
                    event_dict[k] = val
            events.append(event_dict)
        cases.append(events)
    batched_cases = []
    if chunk_cases is None or chunk_cases <= 0:
        chunk_cases = 1
    for i in range(0, len(cases), chunk_cases):
        batch_cases = cases[i:i + chunk_cases]
        batch_events = [e for case in batch_cases for e in case]
        batched_cases.append(batch_events)
        
    return batched_cases


    
def run_fix2(llm, model_, llm_repetition, finalized_mapping, restored_path, df, sys_prompt, user_prompt_tmpl):
    fix2_results = {}
    for target_name, selected_candidates in finalized_mapping.items():
        print(f"\n  - Starting Fix2 (Final Refinement) for: {target_name}")
        refined_events_for_target = []
        df_t, df_o = split_cases_analysis(df, target_name, selected_candidates)
        target_var_json = analyze_variants(df_t, name="Homonymous_Labels_Cases")
        restored_map = restored_path.get(target_name, {})
        if not restored_map:
            print(f"    [Warning] No restored paths found for {target_name} in res_f1.")
            continue
        for variant in target_var_json.get('variants', []):
            rank_val = str(variant.get('rank')) 
            variant['restored_path'] = restored_map.get(rank_val)
        origin_var_str = json.dumps(analyze_variants(df_o, name="Original_Labels_Cases"), indent=4, ensure_ascii=False)
        target_var_str = json.dumps(target_var_json, indent=4, ensure_ascii=False)
        batched_cases = create_batched_jsons(df_t, target_name)
        for input_case in batched_cases:
            input_case_str = json.dumps(input_case, indent=2, ensure_ascii=False)
            user_prompt = user_prompt_tmpl.format(
                HOMONYM_FIX2_INPUT1 = target_name,
                HOMONYM_FIX2_INPUT2 = selected_candidates,
                HOMONYM_FIX2_INPUT3 = origin_var_str,
                HOMONYM_FIX2_INPUT4 = target_var_str,
                HOMONYM_FIX2_INPUT5 = input_case_str
            )
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            while True:
                response = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
                target_event_ids_str = {str(e['event_id']) for e in input_case if 'homonymous_activity' in e}
                try:
                    if not isinstance(response, dict) or 'response' not in response:
                        continue
                    refined_list = response['response'] # [{'event_id': '28', ...}, ...]
                    if not isinstance(refined_list, list):
                        continue
                    response_event_ids = set()
                    is_data_valid = True
                    for res_event in refined_list:
                        res_eid = str(res_event.get('event_id'))
                        response_event_ids.add(res_eid)
                        original_event = next((e for e in input_case if str(e['event_id']) == res_eid), None)
                        if (not original_event or 
                            'homonymous_activity' not in original_event or 
                            res_event.get('homonymous_activity') != original_event.get('homonymous_activity') or 
                            res_event.get('restored_activity') not in selected_candidates):
                            is_data_valid = False
                            break                    
                    if not is_data_valid: 
                        continue
                    if not target_event_ids_str.issubset(response_event_ids): 
                        continue
                    refined_events_for_target.extend(refined_list)
                    break
                except Exception as e:
                    print(f">> [Error] {e}")
                    continue
            fix2_results[target_name] = refined_events_for_target
    return fix2_results