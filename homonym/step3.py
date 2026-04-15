import json
from util import llm_gen

def run_step3(llm, model_, llm_repetition, homonym_candidates, flow_all, sys_prompt, user_prompt_tmpl):
    """
    Validates potential homonyms by synthesizing lexical meanings and process contexts through LLM inference.
    
    This function performs a "Semantic Fusion" analysis to determine if a target activity (e.g., 'Review') 
    is an umbrella term for a set of more specific activities. It moves beyond simple structural 
    matching by reconstructing the 'True Identity' of each activity—combining its literal name 
    with its specific predecessors and successors in the process.

    The validation logic follows a strict synthesis protocol:
    1. Identity Reconstruction: The LLM defines the "Real-World Action" for both the target and 
       candidate members by merging their labels with their contextual triggers and outputs.
    2. Functional Alignment: It checks if the collective actions of the member activities 
       logically explain the existence of the vague, homonymous target label.
    3. Consensus-based Validation: To mitigate stochastic errors, the function uses an 
       ensemble approach. A candidate group is only accepted if the LLM confirms the 
       homonymous relationship in at least 50% (llm_repetition / 2) of the iterations.

    Args:
        llm: The LLM interface for high-level semantic reasoning and reconstruction.
        model_: String identifier for the model architecture version.
        llm_repetition: Number of iterations per candidate to ensure reasoning stability.
        homonym_candidates: Potential homonym groups identified by structural matching in Step 2.
        flow_all: The complete contextual data (predecessors/successors) for all activities.
        sys_prompt: Instructions defining the Semantic Fusion logic and validation criteria.
        user_prompt_tmpl: A template for synthesizing the target's identity against member candidates.

    Returns:
        dict: A refined dictionary of validated homonyms, where Keys are target labels 
              and Values are lists of semantic-functionally aligned member groups.
    """
    print(f">>> Running Step 3 with {llm_repetition} repetitions...")
    flow_lookup = {item['activity']: item for item in flow_all}
    filtered_homonyms = {}
    for target_name, candidates in homonym_candidates.items():
        target_info = flow_lookup.get(target_name)
        target_data = {
            "homonymous_activity": target_name,
            "predecessors": target_info['predecessors'],
            "successors": target_info['successors']
        }
        for combo in candidates:
            combo_details = []
            for member_name in combo:
                member_info = flow_lookup.get(member_name)
                combo_details.append({
                    "member_activity": member_name,
                    "predecessors": member_info['predecessors'],
                    "successors": member_info['successors']
                })
            user_prompt = user_prompt_tmpl.format(
                HOMONYM_STEP3_INPUT1 = target_name,
                HOMONYM_STEP3_INPUT2 = combo,
                HOMONYM_STEP3_INPUT3 = json.dumps({"target": target_data}, indent=2, ensure_ascii=False),
                HOMONYM_STEP3_INPUT4 = json.dumps({"flow_others": combo_details}, indent=2, ensure_ascii=False)
            )
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            found_count = 0
            for attempt in range(llm_repetition):
                print(f"    Repetition {attempt + 1}/{llm_repetition}", end="\r") # 같은 줄에 업데이트
                res = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
                if res.get('found'):
                    found_count += 1
            if found_count >= llm_repetition/2: #1: 
                if target_name not in filtered_homonyms:
                    filtered_homonyms[target_name] = []
                filtered_homonyms[target_name].append(combo)

    return filtered_homonyms