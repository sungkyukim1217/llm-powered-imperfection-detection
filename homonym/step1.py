import pm4py
import pandas as pd

def run_step1(df, filter_list=None, dfg_thres = 0.00, case_col='case_id', time_col='timestamp', act_col='activity'):
    """
    Identifies potential homonym candidates by analyzing execution frequency and structural flow patterns.
    
    This function detects activities that might represent different business steps despite 
    sharing the same label (Homonyms). It employs two primary heuristic filters:
    1. Structural Complexity: It isolates activities with multiple predecessors (>= 2), 
       suggesting a point in the process where various paths converge, which often indicates 
       potential semantic overlap or multi-instance behavior.
    2. Multi-instance Occurrence: It identifies activities that appear two or more times 
       within a single case. This re-occurrence is a strong indicator of a 'Homonymous Label' 
       that may need to be split into distinct context-specific labels.

    The function utilizes PM4Py's Directly-Follows Graph (DFG) to extract the local context 
    (predecessors and successors) for each activity, providing a behavioral baseline for 
    validation in subsequent steps.

    Args:
        df: The pandas DataFrame representing the event log.
        filter_list: An optional set of activities to restrict the analysis to.
        dfg_thres: Frequency threshold for filtering rare process transitions.
        case_col: Column name representing Case IDs.
        time_col: Column name representing event timestamps.
        act_col: Column name representing activity labels.

    Returns:
        tuple: (flow_data_list, flow_data_list_filtered)
            - flow_data_list: Complete list of contextual data for all analyzed activities.
            - flow_data_list_filtered: Refined list of activities identified as homonym candidates.
    """

    
    print(">>> Running Step 1")
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
    flow_data_list_filtered = [] 
    for act in all_activities:
        if filter_list is not None and act not in filter_list:
            continue
        pred_list, succ_list = get_activity_context(act, dfg)
        context_item = {
            'activity': act,
            'predecessors': pred_list,
            'successors': succ_list
        }
        flow_data_list.append(context_item)
        if len(pred_list) >= 2:
            flow_data_list_filtered.append(context_item)
    counts = df.groupby([case_col, act_col]).size().reset_index(name='count')
    activities_appearing_twice = counts[counts['count'] >= 2][act_col].unique()
    flow_data_list_filtered = [
        item for item in flow_data_list_filtered 
        if item['activity'] in activities_appearing_twice
    ]    
    print(f"    - Total activities: {len(flow_data_list)}")
    print(f"    - Potential homonym candidates: {len(flow_data_list_filtered)}")

    return flow_data_list, flow_data_list_filtered

     