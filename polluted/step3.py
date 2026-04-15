from util import llm_gen
from collections import Counter
from collections import defaultdict
import json

import json
from collections import defaultdict
from util import llm_gen

def run_step3(llm, model_, llm_repetition, step2_json, sys_prompt, user_prompt_tmpl):

    """
    Identifies canonical 'Clean Labels' and maps their 'Polluted Variants' using pattern matching and ensemble voting.
    
    This function detects activities contaminated with mutable qualifiers (IDs, codes, timestamps) by 
    evaluating two strict conditions defined in the prompts:
    1. Contextual Similarity: Ensuring the clean label and its variants occupy the same topological 
       position in the process flow (matching summarized predecessors and successors).
    2. Textual Containment (Root Check): Verifying that the clean label serves as the shortest 
       base string (root) and the variants follow the [Root] + [Delimiter] + [Noise] pattern.

    To resolve inconsistencies in LLM-based pattern recognition, the function employs an 
    Ensemble Co-occurrence strategy:
    - Voting: The LLM generates clean-variant mappings 'llm_repetition' times. Each pair 
      (Clean-Variant or Variant-Variant) identified in a mapping receives a vote.
    - Thresholding: Only pairs that consistently appear together in at least 50% of the 
      iterations are considered valid connections.
    - Disjoint Set Union (DSU): A Union-Find algorithm merges stable pairs into final clusters 
      to ensure transitivity and structural integrity.
    - Final Designation: Within each final cluster, the shortest string is designated as the 
      canonical 'Clean Label', and all other members are mapped as its 'Polluted Variants'.

    Args:
        llm: The LLM interface for pattern recognition and semantic validation.
        model_: String identifier for the model version.
        llm_repetition: Number of iterations for ensemble polling to ensure statistical stability.
        step2_json: Summarized contextual data from Step 2.
        sys_prompt: Strategic instructions defining detection criteria for polluted labels and noise patterns.
        user_prompt_tmpl: A template enforcing root identification and delimiter-based variant mapping.

    Returns:
        dict: A finalized mapping dictionary where Keys are Clean Labels (canonical forms) 
              and Values are lists of their corresponding Polluted Variants.
    """
    
    print(f">>> Running Step 3  with {llm_repetition} repetitions...")

    input_data = json.loads(step2_json)
    all_activities = sorted([item['activity'] for item in input_data])
    user_prompt = user_prompt_tmpl.format(POLLUTED_STEP3_INPUT=step2_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    co_occurrence = defaultdict(int)
    for i in range(llm_repetition):
        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        for clean, variants in result.items():
            cluster = [clean] + variants
            for j in range(len(cluster)):
                for k in range(j + 1, len(cluster)):
                    a, b = sorted([cluster[j], cluster[k]])
                    co_occurrence[(a, b)] += 1
    threshold = llm_repetition / 2
    parent = {act: act for act in all_activities}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i]); return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    for (a, b), count in co_occurrence.items():
        if count >= threshold:
            union(a, b)
    clusters = defaultdict(list)
    for act in all_activities:
        clusters[find(act)].append(act)
    final_mapping = {}
    for root, members in clusters.items():
        if len(members) < 2: continue
        clean_label = min(members, key=len)
        variants = [m for m in members if m != clean_label]
        if variants:
            final_mapping[clean_label] = sorted(variants)
    return final_mapping