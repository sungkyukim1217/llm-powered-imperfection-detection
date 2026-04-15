from util import llm_gen

def run_step1(llm, model_, llm_repetition, activity_list_json, sys_prompt, user_prompt_tmpl):

    """
    Identifies 'Polluted Label' candidates by isolating labels with mutable qualifiers from clean templates.
    
    This function targets process mining imperfections where immutable boiler-plate text is 
    contaminated with mutable data such as long numeric IDs (8+ digits), alphanumeric codes, 
    or dates. Based on the provided System Prompt, it distinguishes these 'Polluted Labels' 
    from 'Distorted' (typos) or 'Synonymous' (semantic) labels. 
    
    The filtering logic is designed to:
    1. Extract labels containing variable identifiers (IDs, complex codes, delimiters).
    2. Retain 'Clean' boiler-plate labels ONLY if a corresponding polluted variant exists 
       in the list (ensuring a template-variant relationship is present).
    3. Discard isolated clean labels or unrelated noise patterns.

    An ensemble polling strategy (llm_repetition) is used to maximize the discovery of 
    all polluted patterns, aggregating results into a unique set to ensure high recall 
    before moving to the clustering stage.

    Args:
        llm: The LLM interface for executing natural language inference and filtering.
        model_: String representing the model architecture version.
        llm_repetition: Integer count for redundant polling to ensure comprehensive data capture.
        activity_list_json: A serialized JSON string of raw activity labels to be analyzed.
        sys_prompt: Strategic instructions defining Polluted (Mutable), Distorted (Noise), and Synonymous patterns.
        user_prompt_tmpl: A template enforcing strict filtering to keep only polluted-variant pairs.

    Returns:
        dict: A structured summary containing:
            - "found" (bool): Global flag indicating if any polluted patterns were detected.
            - "data" (list): A lexically sorted list of unique activity names identified as candidates.
    """
    
    print(f">>> Running Step 1  with {llm_repetition} repetitions...")

    user_prompt = user_prompt_tmpl.format(POLLUTED_STEP1_INPUT=activity_list_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    combined_data = set() 
    any_found = False     
    for i in range(llm_repetition):
        print(f"    Repetition {i + 1}/{llm_repetition}", end="\r")
        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        if result.get('found'):
            any_found = True
            combined_data.update(result.get('data', []))
    final_result = {
        "found": any_found,
        "data": sorted(list(combined_data)) 
    }
    return final_result