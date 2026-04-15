from util import llm_gen

def run_step1(llm, model_, llm_repetition, activity_list_json, sys_prompt, user_prompt_tmpl):
    """
    Identifies synonymous activity candidates by filtering out noise patterns and isolating semantic equivalents.
    
    This function utilizes a system prompt defined with a knowledge base of process mining imperfection 
    patterns—specifically distinguishing 'Polluted Labels' (IDs/codes), 'Distorted Labels' (typos/OCR errors), 
    and 'Synonymous Labels' (semantic equivalence). It executes an ensemble polling strategy (llm_repetition) 
     to strictly filter the raw activity list based on ontology rules such as linguistic synonyms, 
    phrase variations, and grammatical transformations. By aggregating results into a unique set, 
    it ensures high recall in capturing all potential synonym groups while discarding isolated 
    or purely noisy labels.

    Args:
        llm: The LLM interface for executing natural language inference tasks.
        model_: Specific identifier for the model architecture (e.g., GPT-4, Gemini-Pro).
        llm_repetition: Integer count for redundant polling to ensure high data coverage/recall.
        activity_list_json: A serialized JSON string of the raw event log activities.
        sys_prompt: Instructions containing definitions for Polluted, Distorted, and Synonymous patterns.
        user_prompt_tmpl: A template enforcing strict filtering logic to keep only semantic synonym pairs.

    Returns:
        dict: A structured summary containing:
            - "found" (bool): Global flag indicating if any synonymous pairs were detected.
            - "data" (list): A lexically sorted list of unique activity names identified as synonym candidates.
    """
    print(f">>> Running Step 1  with {llm_repetition} repetitions...")

    user_prompt = user_prompt_tmpl.format(SYNONYM_STEP1_INPUT=activity_list_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    combined_data = set() 
    any_found = False     
    for i in range(llm_repetition):
        print(f"    Repetition {i + 1}/{llm_repetition}", end="\r") # 같은 줄에 업데이트
        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        if result.get('found'):
            any_found = True
            combined_data.update(result.get('data', []))
    final_result = {
        "found": any_found,
        "data": sorted(list(combined_data)) 
    }
    return final_result