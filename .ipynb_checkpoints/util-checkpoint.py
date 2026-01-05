import json
import os
import pandas as pd
import torch

from openai import OpenAI
from threading import Thread
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


def build_event_jsons_depreciated(
    log_name: str,
    cols_map: dict | None = None,
    use_cols: list[str] | None = None,
    chunk_cases : int | None = 1
) -> tuple[pd.DataFrame, list[list[dict]]]:

    
    df = pd.read_csv(log_name)
    cols_map = {
        "Case": "case_id",
        "Activity": "activity",
        "Timestamp": "timestamp",
        "label": "label",
    }
    use_cols = ["event_id", "case_id", "timestamp", "activity", "label"]
    input_cols = [c for c in use_cols if c not in {"label"}]
    df_new = df[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_new["case_id"] = df_new["case_id"].astype(str)
    df_new.insert(0, "event_id", pd.RangeIndex(len(df_new)).astype(str))
    df_new = df_new[use_cols].copy()
    cases = []
    for case_id, g in df_new.groupby("case_id"):
        events = []
        for _, row in g.iterrows():
            e = {k: str(row[k]) for k in input_cols}  # event_id/case_id 포함 전부 str
            events.append(e)
        cases.append(events)
    batched_cases = []
    if chunk_cases is None or chunk_cases <= 0:
        chunk_cases = 1

    for i in range(0, len(cases), chunk_cases):
        batch_cases = cases[i:i + chunk_cases]
        batch_events = [e for case in batch_cases for e in case]
        batched_cases.append(batch_events)
    return df_new, batched_cases

import pandas as pd

def build_event_jsons(
    log_name: str,
    cols_map: dict | None = None,
    use_cols: list[str] | None = None,
    chunk_cases : int | None = 1
) -> tuple[pd.DataFrame, list[list[dict]]]:

    df = pd.read_csv(log_name)
    if cols_map is None:
        cols_map = {
            "Case": "case_id",
            "Activity": "activity",
            "Timestamp": "timestamp",
            "label": "label",
        }
    if use_cols is None:
        use_cols = ["event_id", "case_id", "timestamp", "activity", "label"]
    df_new = df[list(cols_map.keys())].rename(columns=cols_map).copy()
    df_new.insert(0, "event_id", pd.RangeIndex(len(df_new)))
    df_new["case_id"] = pd.to_numeric(df_new["case_id"], errors='coerce')
    df_new = df_new.sort_values(by=["case_id", "event_id"], ascending=[True, True])
    input_cols = [c for c in use_cols if c not in {"label"}]
    cases = []
    for case_id, g in df_new.groupby("case_id", sort=False):
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
    df_new["case_id"] = df_new["case_id"].astype(str)
    df_new["event_id"] = df_new["event_id"].astype(str)
    df_new = df_new[use_cols].copy()
    return df_new, batched_cases
    
def llm_call(
    model_version: str,
    quantization: bool = True,
    load_in_8bit: bool = True,
    api_key: str = None,
):
    quant_config = None
    if quantization:
        quant_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

    if model_version in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ]:
        model_kwargs = {"torch_dtype": torch.float16}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        pipe = pipeline(
            "text-generation",
            model=model_version,
            device_map="auto",
            model_kwargs=model_kwargs,
        )
        return pipe

    elif model_version == "Qwen/Qwen3-8B":
        tokenizer = AutoTokenizer.from_pretrained(model_version)

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        model = AutoModelForCausalLM.from_pretrained(model_version, **model_kwargs)
        return [tokenizer, model]
    elif model_version in [
            "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o",
        ]:
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
    if model_version in [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ]:
        out = model_instance(
            prompt,
            max_new_tokens=50000,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False,
        )
        text = out[0]["generated_text"]
        if not reasoning:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise ValueError("JSON not found in model output")
            text = text[start:end+1].copy()
        obj = json.loads(text)
        
    elif model_version == "Qwen/Qwen3-8B":
        tokenizer = [model_instance][0]
        model = [model_instance][1]
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=reasoning
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        #When enable_thinking = True
        #try:
        #    index = len(output_ids) - output_ids[::-1].index(151668)
        #except ValueError:
        #    index = 0
        #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        #content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        #print("thinking content:", thinking_content)
        #print("content:", content)
        obj = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
    elif model_version in [
        "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o",
    ]:

        response = model_instance.chat.completions.create(
            model= model_version,
            messages= prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        
        obj = json.loads(response.choices[0].message.content)

    return obj

