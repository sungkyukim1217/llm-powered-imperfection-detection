import json
from collections import defaultdict, Counter
from util import llm_gen


def build_overlap_map(distorted_map):
    """
    Builds the 'overlap_map' from a distorted_map.

    A 'noise word' is overlapping if it appears in the variant list of TWO OR MORE
    distinct clean labels. Such a word is ambiguous and must be adjudicated.

    Args:
        distorted_map: {clean_label: [variant_1, variant_2, ...], ...} produced by Step 2.

    Returns:
        dict: {noise_word: [clean_label_1, clean_label_2, ...], ...}
              containing ONLY the noise words mapped to 2+ clean labels.
    """
    word_to_cleans = defaultdict(list)
    for clean_label, variants in distorted_map.items():
        for var in variants:
            word_to_cleans[var].append(clean_label)
    return {
        word: sorted(cleans)
        for word, cleans in word_to_cleans.items()
        if len(cleans) >= 2
    }


def apply_resolution(distorted_map, resolution):
    """
    Applies the adjudication results to distorted_map so that every previously
    overlapping noise word belongs to AT MOST ONE clean label.

    For each noise word in `resolution`:
      - if resolved to a clean label, keep it ONLY under that clean label;
      - if resolved to None, drop it from ALL clean labels' variant lists.
    Non-overlapping variants are left untouched. Clean labels left empty are dropped.
    """
    resolved_map = {}
    removed_count = 0
    for clean_label, variants in distorted_map.items():
        kept = []
        for var in variants:
            if var in resolution:
                if resolution[var] == clean_label:
                    kept.append(var)          # the adjudicated owner -> keep
                else:
                    removed_count += 1        # wrong owner or None -> drop
            else:
                kept.append(var)              # non-overlapping -> keep
        if kept:
            resolved_map[clean_label] = sorted(kept)
    print(f">>> Resolution applied: removed {removed_count} conflicting entr(ies).")
    return resolved_map


def run_step3(llm, model_, llm_repetition, distorted_map, sys_prompt, user_prompt_tmpl):
    """
    Resolves conflicting noise words that Step 2 mapped to multiple clean labels,
    and returns a conflict-free distorted_map.

    Pipeline:
      1. Detect overlapping noise words (build_overlap_map).
      2. For each one, ask the LLM (ensemble voting) which single clean label it
         truly belongs to, or None.
      3. Apply the verdicts to distorted_map so each noise word belongs to at most
         one clean label (apply_resolution).

    Args:
        llm: The LLM interface.
        model_: String identifier for the model version.
        llm_repetition: Number of iterations per noise word for majority voting.
        distorted_map: {clean_label: [variants]} mapping produced by Step 2.
        sys_prompt: SYSTEM_PROMPT_DISTORTED_STEP3.
        user_prompt_tmpl: USER_PROMPT_DISTORTED_STEP3 template.

    Returns:
        dict: a conflict-free distorted_map ({clean_label: [variants]}).
    """
    print(f">>> Running Step 3  with {llm_repetition} repetitions...")

    overlap_map = build_overlap_map(distorted_map)
    if not overlap_map:
        print(">>> No overlapping noise words found. Step 3 skipped.")
        return {k: sorted(v) for k, v in distorted_map.items()}
    print(f">>> {len(overlap_map)} overlapping noise word(s) to adjudicate.")

    resolution = {}
    for noise_word, candidate_labels in overlap_map.items():
        user_prompt = user_prompt_tmpl.format(
            DISTORTED_STEP3_INPUT1=noise_word,
            DISTORTED_STEP3_INPUT2=json.dumps(candidate_labels, ensure_ascii=False)
        )
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        votes = Counter()
        for i in range(llm_repetition):
            result = llm_gen(model_version=model_, model_instance=llm, prompt=prompt)
            if not result or not isinstance(result, dict):
                continue
            chosen = result.get(noise_word, None)
            # Accept the verdict only if it is a real candidate; otherwise count as None.
            if chosen in candidate_labels:
                votes[chosen] += 1
            else:
                votes[None] += 1

        if votes:
            best_choice, count = votes.most_common(1)[0]
            resolution[noise_word] = best_choice
            print(f"  - '{noise_word}' -> {best_choice} ({count}/{llm_repetition} votes)")
        else:
            resolution[noise_word] = None
            print(f"  - '{noise_word}' -> None (no valid response)")

    # Apply the adjudication results and return the cleaned map.
    final_distorted_map = apply_resolution(distorted_map, resolution)
    return final_distorted_map