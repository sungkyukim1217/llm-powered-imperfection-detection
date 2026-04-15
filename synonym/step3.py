from util import llm_gen
from collections import Counter
from collections import defaultdict
import json

def run_step3(llm, model_, llm_repetition, step2_json, sys_prompt, user_prompt_tmpl):
    """
    Groups activities into synonym clusters using a fuzzy context-matching logic and an ensemble-based voting mechanism.
    
    This function addresses the complexity of semantic clustering by considering two primary factors defined in the prompts:
    1. Contextual Similarity: Comparing summarized predecessor/successor phases to identify functional equivalence in the process flow.
    2. Synonym Boost: Increasing lenience in context matching if the activity labels themselves are strong linguistic synonyms.
    
    To overcome the instability of LLM clustering and ensure transitive consistency, the function implements a Co-occurrence Ensemble strategy:
    - Voting: The LLM performs clustering 'llm_repetition' times. Each pair of activities found in the same cluster receives a vote.
    - Thresholding: Only pairs that appear together in at least 50% (llm_repetition / 2) of the iterations are considered stable.
    - Disjoint Set Union (DSU): A Union-Find algorithm is applied to the stable pairs to build final, mathematically sound clusters, ensuring transitivity (if A=B and B=C, then A=C).

    Args:
        llm: The LLM interface for executing semantic reasoning and clustering.
        model_: String representing the model architecture version.
        llm_repetition: Number of iterations for the ensemble voting to ensure statistical reliability.
        step2_json: Summarized contextual data from Step 2 (strings of business phases).
        sys_prompt: Strategic instructions defining the "Synonym Boost" and "Context Similarity" decision matrix.
        user_prompt_tmpl: Template enforcing fuzzy interpretation of business stages and transitive grouping.

    Returns:
        list: A list of lists, where each sub-list contains activity names that represent the same business process step.
    """

    print(f">>> Running Step 3  with {llm_repetition} repetitions...")

    input_data = json.loads(step2_json)
    all_activities = sorted([item['activity'] for item in input_data])
    user_prompt = user_prompt_tmpl.format(SYNONYM_STEP3_INPUT=step2_json)
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_prompt}]
    co_occurrence = defaultdict(int)
    for i in range(llm_repetition):
        print(f"    Repetition {i + 1}/{llm_repetition}", end="\r") # 같은 줄에 업데이트

        result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
        clusters = result.get('clusters', [])
        for cluster in clusters:
            for j in range(len(cluster)):
                for k in range(j + 1, len(cluster)):
                    a, b = sorted([cluster[j], cluster[k]])
                    co_occurrence[(a, b)] += 1
    threshold = llm_repetition / 2 
    final_pairs = [pair for pair, count in co_occurrence.items() if count >= threshold]
    parent = {act: act for act in all_activities}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i]) 
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
    for a, b in final_pairs:
        union(a, b)
    final_clusters_dict = defaultdict(list)
    for act in all_activities:
        root = find(act) 
        final_clusters_dict[root].append(act)
    return list(final_clusters_dict.values())