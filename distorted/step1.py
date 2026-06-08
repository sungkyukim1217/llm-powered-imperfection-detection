from util import llm_gen

def run_step1(llm, model_, llm_repetition, activity_list_json, sys_prompt, user_prompt_tmpl):
    """
    Identifies canonical 'Clean Labels' from clusters of distorted activities using frequency and edit-distance logic.
    
    This function detects character-level corruptions, such as typos (omissions, insertions, 
    transpositions) and case mutations, by analyzing the raw activity list and its frequencies. 
    Unlike synonyms, these distortions are treated as "noise" that must be mapped back to a 
    single canonical form. 

    The filtering logic follows strict criteria defined in the prompts:
    1. Distorted Cluster Detection: Grouping labels that differ by only one character or 
       capitalization, while actively ignoring acronym biases (e.g., treating 'CHCEK' as a typo).
    2. Canonical Selection (Frequency Priority): For case mutations (e.g., 'login' vs 'Login'), 
       the label with the highest frequency is selected as the canonical form, regardless 
       of grammatical properness.
    3. Spelling Correction: For clear typos, the linguistically correct version within the 
       cluster is preferred as the clean representative.

    An ensemble polling strategy (llm_repetition) ensures consistent identification of these 
    clusters across multiple passes, aggregating the selected canonical labels into a stable 
    candidate set for further processing.

    Args:
        llm: The LLM interface for executing linguistic analysis and frequency-based reasoning.
        model_: String identifier for the model architecture version.
        llm_repetition: Number of iterations to ensure high reliability in cluster identification.
        activity_list_json: A serialized JSON string of activity labels and their occurrence counts.
        sys_prompt: Strategic instructions defining Case Mutation, Omission, Insertion, and Transposition patterns.
        user_prompt_tmpl: A template enforcing frequency-based selection and exact casing preservation.

    Returns:
        dict: A structured summary containing:
            - "found" (bool): Global flag indicating if any distorted clusters were identified.
            - "original_activity" (list): A sorted list of unique labels designated as canonical 'Clean' representatives.
    """
    
    print(f">>> Running Step 1  with {llm_repetition} repetitions...")

    user_prompt = user_prompt_tmpl.format(DISTORTED_STEP1_INPUT=activity_list_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    combined_data = set() 
    for i in range(llm_repetition):
        print(f"    Repetition {i + 1}/{llm_repetition}", end="\r")

        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        if result.get('found'):
            combined_data.update(result.get('original_activity', []))
    clean_activities = sorted(list(combined_data))      
    return clean_activities

