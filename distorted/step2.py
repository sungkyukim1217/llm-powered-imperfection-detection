import json
from collections import defaultdict
from util import llm_gen

def run_step2(llm, model_, llm_repetition, clean_labels, act_freq_dict_json, sys_prompt, user_prompt_tmpl):
    """
    Maps distorted activity variants to their canonical target labels using strict lexical criteria and voting.
    
    This function performs a targeted search for typographical errors associated with each 
    pre-identified 'Clean Label'. It leverages a detailed set of Distortion Criteria, including:
    1. Case Mutation: Variations in capitalization.
    2. Character Omission/Insertion: Single-character differences.
    3. Character Transposition: Swapping of adjacent characters.
    4. Keyboard Proximity: Substitutions based on common keyboard layout errors.

    To ensure the highest precision and avoid false positives (such as synonyms or 
    sub-processes), the function employs an Ensemble Voting mechanism:
    - Target Iteration: Processes each canonical label individually to focus the LLM's attention.
    - Pair Voting: Runs 'llm_repetition' iterations for each target. Every variant proposed 
      by the LLM is recorded as a 'vote' for the (Target, Variant) pair.
    - Threshold Validation: Only variants that appear in at least 50% of the iterations 
      are accepted into the final map, filtering out inconsistent LLM hallucinations.

    Args:
        llm: The LLM interface for executing string distance and typographical analysis.
        model_: String identifier for the model version.
        llm_repetition: Number of iterations per target to ensure statistical stability.
        clean_labels: A list of canonical labels identified in Step 1.
        act_freq_dict_json: A JSON-serialized dictionary of all activities and their counts.
        sys_prompt: Strategic instructions defining strict lexical and exclusion criteria.
        user_prompt_tmpl: A template for mapping specific canonical targets to their variants.

    Returns:
        dict: A mapping dictionary where Keys are Clean Labels and Values are sorted lists 
              of their confirmed distorted variants.
    """
    final_distorted_map = {}

    print(f">>> Running Step 2  with {llm_repetition} repetitions...")

    for target_act in clean_labels:
        pair_votes = defaultdict(int)
        
        user_prompt = user_prompt_tmpl.format(
            DISTORTED_STEP2_INPUT1=target_act,
            DISTORTED_STEP2_INPUT2=act_freq_dict_json
        )
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        for i in range(llm_repetition):
            result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
            if result and isinstance(result, dict):
                variants = result.get(target_act, [])
                for var in variants:
                    pair_votes[var] += 1
        
        threshold = llm_repetition / 2
        final_variants = [var for var, count in pair_votes.items() if count >= threshold]

        if final_variants:
            final_distorted_map[target_act] = sorted(final_variants)

    return final_distorted_map