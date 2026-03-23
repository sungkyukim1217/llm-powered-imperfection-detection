import argparse
import json
import os
import re
import pandas as pd
import torch
import numpy as np
from itertools import combinations
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from threading import Thread
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import pm4py
from util import *

load_dotenv()
pd.set_option('display.max_rows', None)


def homonym_step1(df: pd.DataFrame, case_col: str = 'case_id', time_col: str = 'timestamp', act_col: str = 'activity', filter_list: set = None):
    df_pm4py = df[[case_col, time_col, act_col]].copy()
    df_pm4py.rename(columns={
        case_col: "case:concept:name",
        time_col: "time:timestamp",
        act_col: "concept:name"
    }, inplace=True)
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"], errors="coerce")
    heu_net = pm4py.discover_heuristics_net(df_pm4py)
    
    temp_preds = defaultdict(list)
    temp_succs = defaultdict(list)
    for (src, dst), freq in heu_net.dfg.items():
        temp_succs[src].append((dst, freq))
        temp_preds[dst].append((src, freq))
        
    flow_data_list = []
    flow_data_list_filtered = [] 
    all_activities = sorted(df[act_col].unique())
    
    for act in all_activities:
        if filter_list is not None and act not in filter_list:
            continue
        pred_list = [item[0] for item in sorted(temp_preds[act], key=lambda x: x[1], reverse=True)]
        succ_list = [item[0] for item in sorted(temp_succs[act], key=lambda x: x[1], reverse=True)]
        
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
    flow_data_list_final = [item for item in flow_data_list_filtered if item['activity'] in activities_appearing_twice]
    
    return flow_data_list, flow_data_list_final

def homonym_step2(target_activity, flow_others):
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
            if len(pre_intersection) < 2 or len(suc_intersection) < 1:
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
    if len(sorted_results) <= 5:
        return sorted_results
        
    fifth_error_val = sorted_results[4]['error_count']
    filtered_results = [res for res in sorted_results if res['error_count'] <= fifth_error_val]
    return filtered_results

SYSTEM_PROMPT_HOMONYM_STEP3 = """
You are a Process Mining Expert specializing in Semantic Reconstruction.
Your goal is to validate if a group of activities is a 'Homonymous Decomposition' of a Target Activity by synthesizing their literal names and process contexts.

### CORE LOGIC: THE SEMANTIC FUSION
Do not treat names and contexts separately. You must RECONSTRUCT the 'True Identity' of each activity as follows:
1. LEXICAL MEANING: What does the activity name (e.g., 'Review') imply in a general business sense?
2. CONTEXTUAL MEANING: What do the Predecessors (triggers) and Successors (outputs) reveal about its specific role in this process?
3. SYNTHESIS (THE TRUE SEMANTIC): Combine 1 & 2 to define the "Real-World Action" being performed.

### VALIDATION CRITERIA
- RECONSTRUCTION MATCH: Does the 'True Semantic Identity' of the Target Activity encompass the 'Combined True Semantic Identities' of ALL members in the candidate group?
- AMBIGUITY RESOLUTION: Is the Target name a vague "umbrella term" that effectively describes the specific functional roles revealed by the members' contexts?

### STRICT CONSTRAINTS
- ALL-OR-NOTHING: Evaluate the group as a whole. Do not modify the list.
- NO PROSE: Output ONLY the JSON object.
"""


def get_homonym_user_prompt_step3(target_activity_json, flow_others_json):
    target_data = json.loads(target_activity_json)
    flow_others_data = json.loads(flow_others_json)
    target_name = target_data['target']['homonymous_activity']
    input_members = [m['member_activity'] for m in flow_others_data['flow_others']]
    return f"""
### TASK: Comprehensive Semantic Validation

**OBJECTIVE:**
Analyze if the set {input_members} is the specific realization of the homonymous activity "{target_name}".

**INPUT DATA:**
1. [TARGET ACTIVITY]: {target_activity_json}
2. [FLOW OTHERS] (Candidates to validate): {flow_others_json}

**EXECUTION STEPS:**
1. **Target Identity Reconstruction:** Combine the name "{target_name}" with its Pre/Suc. Define exactly what "True Action" this activity represents here.
2. **Member Identity Reconstruction:** For each activity in {input_members}, combine its name with its Pre/Suc. Define the "True Action" of each member.
3. **Synthesis & Comparison:** - Does the collective "True Action" of these members explain why they might have been incorrectly grouped under the name "{target_name}"?
   - Is there a functional alignment between the Target's context and the Members' combined context?

**STRICT OUTPUT FORMAT:**
- If Match: {{ "found": true, "data": {{ "homonymous_label": "{target_name}", "member_activities": {input_members} }} }}
- If No Match: {{ "found": false, "data": [] }}
"""

def get_heuristics_summary(df: pd.DataFrame, case_col: str = 'case_id', time_col: str = 'timestamp', act_col: str = 'activity'):
    df_pm4py = df[[case_col, time_col, act_col]].rename(columns={
        case_col: "case:concept:name",
        time_col: "time:timestamp",
        act_col: "concept:name"
    })
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"])
    heu_net = pm4py.discover_heuristics_net(df_pm4py)
    return "\n".join(f"{src} -> {dst} (Freq: {freq})" for (src, dst), freq in heu_net.dfg.items())

SYSTEM_PROMPT_HOMONYM_STEP4 = """
You are a Process Mining Expert.
Your goal is to evaluate multiple candidate groups at once and select the SINGLE BEST candidate group that represents the 'Target Activity' (Homonymous Label).

### LOGIC
1. Evaluate all provided Candidate Combos against the [HEURISTICS NET SUMMARY].
2. For each combo, look at its members' individual Predecessors and Successors to understand their exact position in the process flow.
3. Select the ONE combo that best aligns with the Target's structural position. The optimal substitution must result in the cleanest, most logically coherent process model, avoiding tangled or illogical routing (spaghetti flows).

### SELECTION CRITERIA
- **Phase Alignment**: Do the candidate members naturally operate in the exact same process phase as the Target?
- **Structural Simplicity**: If the Target is replaced by this combo, does the overall process flow remain clear and straightforward?
- **Behavioral Equivalence**: Does the combo collectively fulfill the exact routing purpose of the Target without introducing illogical detours?

### STRICT CONSTRAINTS (CRITICAL)
- **NO MIX AND MATCH**: You MUST select exactly ONE combo from the provided list. Do NOT pick members from different combos to create a new one.
- **EXACT MATCH**: The array in "original_activity" MUST be exactly identical to the members of the chosen combo.

### OUTPUT FORMAT
Do NOT provide any explanations, reasoning, or markdown blocks.
Output ONLY a JSON object in this exact format:
{
  "homonymous_label": "Target Name",
  "original_activity": ["Member 1 from chosen combo", "Member 2 from chosen combo"]
}
"""

def get_homonym_user_prompt_step4(heu_sum, target_data, all_candidates_data):
    return f"""
### [HEURISTICS NET SUMMARY: THE GROUND TRUTH]
{heu_sum}

### [TARGET ACTIVITY: THE LABEL TO RESOLVE]
- **Name**: "{target_data['activity']}"
- **Predecessors**: {target_data['pre']}
- **Successors**: {target_data['suc']}

### [ALL CANDIDATE COMBOS TO EVALUATE]
Below are all the candidate combos for this target. Each combo details the individual context of its members.
Compare them and pick the single best combo.

{json.dumps(all_candidates_data, indent=2, ensure_ascii=False)}

### [YOUR TASK]
Return the single best combo in the requested JSON format. NO PROSE.
"""

def main():
    parser = argparse.ArgumentParser(description="Homonym Resolution Script")
    parser.add_argument('--log_name', type=str, required=True, help="Name of the log file (e.g., pub_seed1_03_homonymous)")
    parser.add_argument('--model', type=str, default="gpt-5.1", help="LLM model version (default: gpt-5.1)")
    parser.add_argument('--chunk_size', type=int, default=1, help="Chunk size for cases (default: 1)")
    args = parser.parse_args()

    model_ = args.model
    log_name_ = args.log_name
    chunk_size_ = args.chunk_size

    llm = llm_call(model_version=model_, api_key=os.getenv("API_KEY"))
    df_new, cases_json = build_event_jsons(log_name=f"./dataset/{log_name_}.csv", chunk_cases=chunk_size_)
    
    flow_all, flow_filtered = homonym_step1(df_new)

    homonym_candidates = {}
    for target in flow_filtered:
        flow_others_ = [item for item in flow_all if item['activity'] != target['activity']]
        matches = homonym_step2(target, flow_others_)
        if matches:
            target_name = target['activity']
            homonym_candidates[target_name] = [m['matched_activities'] for m in matches]

    flow_lookup = {item['activity']: item for item in flow_all}
    final_validated_homonyms = {}
    
    for target_name, candidates in homonym_candidates.items():
        target_info = flow_lookup.get(target_name)
        target_data = {
            "homonymous_activity": target_name,
            "predecessors": target_info['predecessors'],
            "successors": target_info['successors']
        }
        for i, combo in enumerate(candidates, 1):
            combo_details = [{
                "member_activity": member_name,
                "predecessors": flow_lookup.get(member_name)['predecessors'],
                "successors": flow_lookup.get(member_name)['successors']
            } for member_name in combo]
            
            target_json = json.dumps({"target": target_data}, indent=2, ensure_ascii=False)
            flow_others_json = json.dumps({"flow_others": combo_details}, indent=2, ensure_ascii=False)
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT_HOMONYM_STEP3},
                {"role": "user", "content": get_homonym_user_prompt_step3(target_json, flow_others_json)}
            ]
            
            raw_output = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
            
            try:
                res = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
                if res.get('found'):
                    label = res['data']['homonymous_label']
                    members = res['data']['member_activities']
                    if label not in final_validated_homonyms:
                        final_validated_homonyms[label] = []
                    final_validated_homonyms[label].append(members)
            except Exception:
                pass

    heu_sum = get_heuristics_summary(df_new)

    for target_name, candidates in final_validated_homonyms.items():
        target_info = {
            "activity": target_name,
            "pre": flow_lookup[target_name]['predecessors'],
            "suc": flow_lookup[target_name]['successors']
        }
            
        all_combos_details = [
            {
                "combo_id": i,
                "combo_members": combo,
                "member_details": [
                    {
                        "member_name": member,
                        "predecessors": flow_lookup.get(member, {}).get('predecessors', []),
                        "successors": flow_lookup.get(member, {}).get('successors', [])
                    } for member in combo
                ]
            } for i, combo in enumerate(candidates, 1)
        ]
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT_HOMONYM_STEP4},
            {"role": "user", "content": get_homonym_user_prompt_step4(heu_sum, target_info, all_combos_details)}
        ]
        
        response = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)        
        print(response)

if __name__ == "__main__":
    main()