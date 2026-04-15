import json
import os
import pandas as pd
import random
import torch
import pm4py 
from collections import defaultdict, Counter
from openai import OpenAI
from threading import Thread
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from collections import Counter, defaultdict
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter

def build_event_jsons(
    log_name: str,
    case_id : str = 'case_id',
    activity : str = 'activity',
    timestamp : str = 'timestamp',
    event_id : str = 'event_id',
    label : str = 'label',
    cols_map: dict | None = None,
    use_cols: list[str] | None = None,
    chunk_cases : int | None = 1,
    df_pattern : str | None = None
) -> tuple[pd.DataFrame, list[list[dict]]]:

    df = pd.read_csv(log_name)
    if cols_map is None:
        target_label_col = "Injection" if "Injection" in df.columns else "label"
        cols_map = {
            "Case": case_id,
            "Activity": activity,
            "Timestamp": timestamp,
            target_label_col: label,
        }
    if use_cols is None:
        use_cols = [event_id, case_id, activity, timestamp, label]
    df_new = df[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_new.insert(0, event_id, pd.RangeIndex(len(df_new)))
    df_new[case_id] = pd.to_numeric(df_new[case_id], errors='coerce')
    df_new = df_new.sort_values(by=[case_id, event_id], ascending=[True, True])
    input_cols = [c for c in use_cols if c not in {label}]
    cases = []
    for cid, g in df_new.groupby(case_id, sort=False):
        events = []
        for _, row in g.iterrows():
            e = {k: str(row[k]) for k in input_cols}
            events.append(e)
        cases.append(events)
    batched_cases = []
    if chunk_cases is None or chunk_cases <= 0:
        chunk_cases = 1
    for i in range(0, len(cases), chunk_cases):
        batch_cases = cases[i:i + chunk_cases]
        batch_events = [e for case in batch_cases for e in case]
        batched_cases.append(batch_events)
    df_new[case_id] = df_new[case_id].astype(str)
    df_new[event_id] = df_new[event_id].astype(str)
    df_new = df_new[use_cols].copy()
    return df_new, batched_cases

def llm_call(
    model_version: str,
    api_key: str = None,
):
    SUPPORTED_OPENAI_MODELS = [
        "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5.1", "gpt-5.2"
    ]
    if model_version in SUPPORTED_OPENAI_MODELS:
        if api_key is None or not api_key.strip():
            raise ValueError(f"Error: API Key is missing for model '{model_version}'. Please provide a valid 'api_key'.")
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()
        return client

        
    else:
        raise ValueError(f"Unsupported model_version: {model_version}")



def llm_gen(
    model_version: str,
    model_instance,
    prompt: list[dict],
    reasoning: bool = False,
):
    SUPPORTED_OPENAI_MODELS = [
        "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o", "gpt-5.1", "gpt-5.2"
    ]
    if model_version in SUPPORTED_OPENAI_MODELS:
        current_seed = int(random.randint(0, 1000000))
        response = model_instance.chat.completions.create(
            model= model_version,
            messages= prompt,
            temperature=0.0,
            seed = current_seed,
            response_format={"type": "json_object"},
        )
        
        obj = json.loads(response.choices[0].message.content)

        return obj
    else:
        raise ValueError(f"Unsupported model: {model_version}. Available: {SUPPORTED_OPENAI_MODELS}")

