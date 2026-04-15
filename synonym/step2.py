import json
import pm4py
import pandas as pd
from util import llm_gen

def get_synonym_context(df: pd.DataFrame,
                        target_list: set = None,
                        dfg_thres: float = 0.00,
                        case_col: str = 'case_id',
                        time_col: str = 'timestamp',
                        act_col: str = 'activity'
                       ):
    """
    Extracts the local process context (predecessors and successors) for specified activities using DFG.
    
    This function converts a standard DataFrame into a PM4Py-compatible event log to discover 
    Directly-Follows Graphs (DFG). It identifies the immediate flow relations for each activity 
    in the 'target_list'. The resulting context provides a behavioral profile (neighboring activities) 
    that helps the LLM distinguish or group activities based on their execution patterns rather 
    than just their labels.
    
    Args:
        df: The raw event log DataFrame.
        target_list: A set of activities to analyze. If None, all unique activities are processed.
        dfg_thres: Minimum frequency threshold (percentage) to include a flow relation in the context.
        case_col: Column name representing Case IDs.
        time_col: Column name representing Timestamps.
        act_col: Column name representing Activity labels.

    Returns:
        str: A JSON-serialized list of dictionaries containing 'activity', 'predecessors', and 'successors'.
    """
    
    df_pm4py = df[[case_col, time_col, act_col]].copy()
    df_pm4py.rename(columns={
        case_col: "case:concept:name",
        time_col: "time:timestamp",
        act_col: "concept:name"
    }, inplace=True)
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"], errors="coerce")
    dfg, start_activities, end_activities = pm4py.discover_dfg(df_pm4py)
    def get_activity_context(activity, dfg_dict):
        predecessors = {k[0]: v for k, v in dfg_dict.items() if k[1] == activity}
        successors = {k[1]: v for k, v in dfg_dict.items() if k[0] == activity}
        total_pred = sum(predecessors.values())
        total_succ = sum(successors.values())
        def format_to_list(dist_dict, total):
            if total == 0: return []
            items = [(k, v/total) for k, v in dist_dict.items() if (v/total) >= dfg_thres]
            items.sort(key=lambda x: x[1], reverse=True)
            return [k for k, v in items]
        return format_to_list(predecessors, total_pred), format_to_list(successors, total_succ)
    all_activities = sorted(df[act_col].unique())
    flow_data_list = []
    for act in all_activities:
        if target_list is not None and act not in target_list:
            continue
        pred, succ = get_activity_context(act, dfg)
        flow_data_list.append({
            'activity': act,
            'predecessors': pred, # 이제 리스트입니다 ['A', 'B']
            'successors': succ    # 이제 리스트입니다 ['C', 'D']
        })
        
    return json.dumps(flow_data_list, indent=2, ensure_ascii=False)


def run_step2(llm, model_, llm_repetition, context_json, sys_prompt, user_prompt_tmpl):

    """
    Validates and summarizes the structural context of activities using a retry-based LLM validation mechanism.
    
    This function transforms detailed lists of neighbor activities into high-level business phase summaries.
    The underlying prompts enforce a strict Abstraction Logic:
    1. Identifying the 'Core Action' within predecessors/successors lists.
    2. Filtering out noise (typos, minor variations) to find a 'Common Business Phase'.
    3. Replacing exhaustive lists with a single descriptive string (e.g., ["MRI", "X-Ray"] -> "Diagnosis Stage").

    To ensure data integrity, the function employs a Validation Retry Mode:
    - It compares the set of input activities against the LLM's output activities.
    - If the LLM omits any activity during the summarization process (a common LLM behavior with long lists), 
      the function automatically retries up to 'llm_repetition' times.
    - It returns the final result only when all activities are successfully accounted for or the retry limit is hit.

    Args:
        llm: The LLM instance for processing natural language tasks.
        model_: Specific identifier for the model version.
        llm_repetition: Maximum number of retry attempts for structural validation.
        context_json: JSON string from get_synonym_context containing raw flow lists.
        sys_prompt: Strategic instructions defining the Abstraction and Summarization Logic.
        user_prompt_tmpl: A template enforcing the transformation of lists into single summarized strings.

    Returns:
        dict: A dictionary containing the "summarized_context" where each flow is a simplified business phase.
    """
    print(f">>> Running Step 2 (Validation Retry Mode, Max: {llm_repetition})")
    
    input_data = json.loads(context_json)
    input_activities = set(item['activity'] for item in input_data)
    user_prompt = user_prompt_tmpl.format(SYNONYM_STEP2_INPUT=context_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    for attempt in range(llm_repetition):
        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        summarized_list = result.get("summarized_context", [])
        output_activities = set(item['activity'] for item in summarized_list)
        missing_activities = input_activities - output_activities
        if not missing_activities:
            print(f">>> Step 2 success: All {len(input_activities)} activities summarized.")
            return result
        else:
            print(f">>> Step 2 attempt {attempt + 1} failed. Missing: {missing_activities}")
    print(">>> Step 2 warning: Max retries reached with missing items.")
    return result