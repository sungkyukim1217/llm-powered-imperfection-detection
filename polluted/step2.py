import json
import pm4py
import pandas as pd
from util import llm_gen

def get_polluted_context(df: pd.DataFrame,
                        target_list: set = None,
                        dfg_thres: float = 0.00,
                        case_col: str = 'case_id',
                        time_col: str = 'timestamp',
                        act_col: str = 'activity'
                       ):

"""
    Extracts behavioral flow context (predecessors and successors) for potential polluted labels using DFG.
    
    This function utilizes the PM4Py library to discover a Directly-Follows Graph (DFG) from the 
    event log. It specifically isolates the 'input' and 'output' neighbors for each activity 
    identified in the target_list. By providing this structural context, it allows the LLM 
    to verify if various polluted labels (e.g., 'Task_001', 'Task_002') share the same 
    topological position in the process, confirming they are variants of the same clean activity.

    Args:
        df: The pandas DataFrame containing event log data.
        target_list: A set of activity labels (clean and polluted candidates) to be analyzed.
        dfg_thres: Frequency threshold to filter out rare or noisy transitions from the context.
        case_col: The column name for Case IDs.
        time_col: The column name for event timestamps.
        act_col: The column name for activity labels.

    Returns:
        str: A JSON-serialized list of dictionaries representing the local flow environment for each target.
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
    Summarizes detailed flow lists into high-level business phases while ensuring data integrity.
    
    This function processes the raw contextual data of polluted labels through an abstraction 
    logic defined in the prompts. It transforms exhaustive lists of neighbor activities into 
    single, descriptive 'Process Stage Names' (e.g., ["Login_123", "Auth_Retry"] -> "User Authentication").
    This abstraction helps in subsequent steps by providing a standardized semantic baseline 
    to compare different polluted variants.

    The function includes a robust 'Validation Retry Mode':
    1. It compares the activities in the LLM's output against the original input set.
    2. If the LLM omits any activity (a risk when processing multiple polluted variants), 
       the function triggers a retry up to 'llm_repetition' times.
    3. This ensures that every detected polluted candidate is accounted for in the final 
       summarized context, preventing data loss during the pre-processing pipeline.

    Args:
        llm: The LLM interface for semantic summarization and reasoning.
        model_: The specific model architecture identifier.
        llm_repetition: Maximum number of retry attempts allowed for validation success.
        context_json: JSON string of the raw DFG context from get_polluted_context.
        sys_prompt: Strategic instructions for identifying core actions and formulating summaries.
        user_prompt_tmpl: A template enforcing the transformation of lists into single summarized strings.

    Returns:
        dict: A dictionary containing 'summarized_context', where each activity has a string-based flow description.
    """

    
    print(f">>> Running Step 2 (Validation Retry Mode, Max: {llm_repetition})")
    
    input_data = json.loads(context_json)
    input_activities = set(item['activity'] for item in input_data)
    user_prompt = user_prompt_tmpl.format(POLLUTED_STEP2_INPUT=context_json)
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