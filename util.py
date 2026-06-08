import json
import os
import pandas as pd
import random
import torch
import time
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

def prepare_event_df(
    log_name: str,
    case_id: str = 'case_id',
    activity: str = 'activity',
    timestamp: str = 'timestamp',
    event_id: str = 'event_id',
    label: str = 'label',
    cols_map: dict | None = None,
    use_cols: list[str] | None = None
) -> pd.DataFrame:
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
    df_new[case_id] = df_new[case_id].astype(str)
    df_new[event_id] = df_new[event_id].astype(str)
    df_new = df_new[use_cols].copy()
    return df_new



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
    retries: int = 3
):
    SUPPORTED_OPENAI_MODELS = [
        "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o", 
        "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4"
    ]
    
    if model_version not in SUPPORTED_OPENAI_MODELS:
        raise ValueError(f"Unsupported model: {model_version}.")

    attempt = 0
    while attempt < retries:
        try:
            current_seed = int(random.randint(0, 1000000))
            
            # [수정 포인트] prompt 내부의 content에 혹시 모를 제어 문자가 있다면 정화
            # 가끔씩 발생하는 BadRequest 400 에러의 핵심 방어 로직입니다.
            safe_prompt = []
            for m in prompt:
                # 불필요한 제어 문자 제거 (\x00-\x1f 범위 등)
                clean_content = "".join(ch for ch in m['content'] if ord(ch) >= 32 or ch in "\n\r\t")
                safe_prompt.append({"role": m['role'], "content": clean_content})

            response = model_instance.chat.completions.create(
                model=model_version,
                messages=safe_prompt, # 정화된 프롬프트 사용
                temperature=0.0,
                seed=current_seed,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response received from LLM.")

            return json.loads(content)

        except Exception as e:
            attempt += 1
            # "We could not parse the JSON body" 문구가 포함된 경우 재시도 로직 진입
            error_msg = str(e)
            if "JSON body" in error_msg or "400" in error_msg or "rate_limit" in error_msg.lower():
                print(f"!!! [Attempt {attempt}/{retries}] Payload Parsing Error or Timeout: {error_msg[:100]}...")
                time.sleep(2)
                continue
            else:
                print(f"!!! [Critical Error] {e}")
                raise e

    raise Exception(f"Failed to receive a valid response after {retries} attempts.")