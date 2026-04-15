from itertools import combinations

def find_homonym_matches(target_activity, flow_others, candidate_number = 5):
    """
    Finds combinations of other activities that structurally mimic the flow of a target activity.
    
    This function operates on the hypothesis that a 'Homonymous Label' is a mixture of multiple 
    distinct process steps. It uses combinatorial search and set theory to find a set of 
    activities (r=2 to 5) whose unified predecessors and successors closely align with those 
    of the target activity.

    The matching process includes:
    1. Redundancy Filtering: It ensures every activity in a candidate combination contributes 
       unique, relevant flow information to the target. If an activity in the combo only 
       adds 'noise' (extra flows not in target) without adding 'value' (shared flows), 
       the combination is discarded.
    2. Intersection Validation: At least one predecessor and one successor must be shared 
       between the target and the combined candidate set.
    3. Error Scoring (Symmetric Difference): It calculates the 'error_count' using the 
       symmetric difference between sets. A lower error count indicates a higher structural 
       similarity, suggesting the target label is likely a homonym of the matched set.

    Args:
        target_activity: Dictionary containing context ('predecessors', 'successors') of the target.
        flow_others: List of context dictionaries for all other activities in the log.
        candidate_number: The maximum number of top matches (based on error count) to return.

    Returns:
        list: A ranked list of dictionaries containing matched activity sets and their error scores.
    """
    target_pre = set(target_activity['predecessors'])
    target_suc = set(target_activity['successors'])
    target_name = target_activity['activity']
    all_results = [] 
    for r in range(2, 6):
        for combo in combinations(flow_others, r):
            is_redundant_combo = False
            for i, act_item in enumerate(combo):
                others = [x for j, x in enumerate(combo) if i != j]
                union_pre_others = set().union(*[set(x['predecessors']) for x in others])
                union_suc_others = set().union(*[set(x['successors']) for x in others])
                my_pre = set(act_item['predecessors'])
                my_suc = set(act_item['successors'])
                new_correct_pre = (my_pre & target_pre) - union_pre_others
                new_correct_suc = (my_suc & target_suc) - union_suc_others
                my_extra_pre = my_pre - target_pre
                my_extra_suc = my_suc - target_suc
                if (not new_correct_pre and not new_correct_suc) and (my_extra_pre or my_extra_suc):
                    is_redundant_combo = True
                    break
            if is_redundant_combo:
                continue
            combined_pre = set().union(*[set(x['predecessors']) for x in combo])
            combined_suc = set().union(*[set(x['successors']) for x in combo])
            pre_intersection = target_pre & combined_pre
            suc_intersection = target_suc & combined_suc
            if len(pre_intersection) < 1 or len(suc_intersection) < 1:
                continue
            pre_diff = target_pre ^ combined_pre
            suc_diff = target_suc ^ combined_suc
            total_error = len(pre_diff) + len(suc_diff)
            all_results.append({
                "target": target_name,
                "matched_activities": [x['activity'] for x in combo],
                "error_count": total_error
            })
    if not all_results:
        return []
    sorted_results = sorted(all_results, key=lambda x: x['error_count'])
    if len(sorted_results) <= candidate_number:
        return sorted_results
    fifth_error_val = sorted_results[4]['error_count']
    return [res for res in sorted_results if res['error_count'] <= fifth_error_val]

def run_step2(flow_all, flow_filtered):
    """
    Identifies structural homonym candidates by executing combinatorial matching for filtered activities.
    
    This function iterates through the potential candidates identified in Step 1 and 
    attempts to find structural 'twins' or 'components' among the rest of the event log's 
    activities. By comparing the behavioral context of a target against every possible 
    combination of other activities, it flags labels that are likely overloaded with 
    multiple semantic meanings.

    It acts as a bridge between data-driven heuristic filtering (Step 1) and LLM-based 
    semantic validation (Step 3), significantly narrowing down the search space to only 
    those labels that show mathematically plausible homonym patterns.

    Args:
        flow_all: Complete list of contextual data for all activities in the event log.
        flow_filtered: Refined list of activities flagged as potential homonyms in Step 1.

    Returns:
        dict: A dictionary where Keys are homonym candidate labels and Values are lists 
              of possible matching combinations (activity sets).
    """
    print(">>> Running Step 2")
    homonym_candidates = {}
    
    for target in flow_filtered:
        flow_others = [item for item in flow_all if item['activity'] != target['activity']]
        matches = find_homonym_matches(target, flow_others)
        
        if matches:
            target_name = target['activity']
            homonym_candidates[target_name] = [m['matched_activities'] for m in matches]
            print(f"  - Match found for '{target_name}': {len(matches)} combinations")
            
    return homonym_candidates