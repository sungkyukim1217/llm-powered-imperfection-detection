import json
import os
import pandas as pd
import torch

from openai import OpenAI
from threading import Thread
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from collections import Counter, defaultdict
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter

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

def get_dfg_abstraction(df_input):
s    df_pm4py = df_input[['case_id', 'timestamp', 'activity']].copy().rename(columns={
        'case_id': "case:concept:name",
        'timestamp': "time:timestamp",
        'activity': "concept:name"
    })
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"], errors="coerce")
    df_pm4py = dataframe_utils.convert_timestamp_columns_in_df(df_pm4py)
    event_log = log_converter.apply(df_pm4py)
    dfg_freq = dfg_algorithm.apply(event_log, variant=dfg_algorithm.Variants.FREQUENCY)
    dfg_perf = dfg_algorithm.apply(event_log, variant=dfg_algorithm.Variants.PERFORMANCE)
    rows = []
    for (source, target), freq in dfg_freq.items():
        perf = dfg_perf.get((source, target), None)
        perf_str = f"{perf:.2f}" if isinstance(perf, (int, float)) and perf is not None else "NA"
        rows.append((source, target, freq, perf_str))
    rows.sort(key=lambda x: (-x[2], x[0], x[1]))
    formatted_lines = []
    formatted_lines.append("### DIRECTLY-FOLLOWS GRAPH (DFG)")
    formatted_lines.append("This abstraction captures the direct succession of activities (A -> B) to identify process flows and bottlenecks.")
    formatted_lines.append(f"- **Total Transitions**: {len(rows)} unique paths observed.")
    formatted_lines.append("- **Metrics**: 'transition_count' (Frequency strength), 'mean_duration_seconds' (Average time taken between activities).")
    formatted_lines.append("")
    for source, target, freq, perf in rows:
        line = f"{source} -> {target} (transition_count={freq}, mean_duration_seconds={perf})"
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

def get_variant_abstraction(df_input, top_k=50):
    df_processed = df_input[['case_id', 'timestamp', 'activity']].copy().rename(columns={
        'case_id': "case:concept:name",
        'timestamp': "time:timestamp",
        'activity': "concept:name"
    })
    df_processed["time:timestamp"] = pd.to_datetime(df_processed["time:timestamp"], errors="coerce")
    case_to_seq = df_processed.groupby("case:concept:name")["concept:name"].apply(list)
    case_times = (
        df_processed.groupby("case:concept:name")["time:timestamp"]
        .agg(lambda s: (s.max() - s.min()).total_seconds())
    )
    variant_counter = Counter()
    variant_durations = defaultdict(list)
    for cid, seq in case_to_seq.items():
        var = tuple(seq)
        variant_counter[var] += 1
        variant_durations[var].append(case_times.loc[cid])
    records = []
    for v, freq in variant_counter.items():
        durs = variant_durations[v]
        avg_perf = (sum(durs) / len(durs)) if durs else None
        records.append({"variant": v, "case_count": freq, "mean_case_duration_seconds": avg_perf})
    res = pd.DataFrame(records).sort_values(["case_count", "mean_case_duration_seconds"], ascending=[False, True])
    total_cases = res["case_count"].sum()
    total_variants = len(res)
    res_top = res.head(top_k)
    shown_cases = res_top["case_count"].sum()
    coverage = (shown_cases / total_cases) * 100 if total_cases > 0 else 0
    formatted_lines = []
    formatted_lines.append(f"### PROCESS VARIANTS (Top {len(res_top)})")
    formatted_lines.append("This list represents the most frequent activity sequences (paths) found in the event log.")
    formatted_lines.append(f"- **Summary**: Showing top {len(res_top)} out of {total_variants} unique variants.")
    formatted_lines.append(f"- **Coverage**: These variants cover **{shown_cases}** out of **{total_cases}** total cases (**{coverage:.1f}%**).")
    formatted_lines.append("")
    for _, row in res_top.iterrows():
        variant_seq = row["variant"]
        if isinstance(variant_seq, (tuple, list)):
            v_str = " -> ".join(str(v) for v in variant_seq)
        else:
            v_str = str(variant_seq)
            
        freq = int(row["case_count"])
        perf = row["mean_case_duration_seconds"]
        
        if pd.notnull(perf):
            perf_str = f"{perf:.2f}"
        else:
            perf_str = "NA"
        line = f"{v_str} (case_count={freq}, mean_case_duration_seconds={perf_str})"
        formatted_lines.append(line)
    return "\n".join(formatted_lines)
    
def get_petri_net_abstraction(df_input):
    df_pm4py = df_input[['case_id', 'timestamp', 'activity']].copy().rename(columns={
        'case_id': "case:concept:name",
        'timestamp': "time:timestamp",
        'activity': "concept:name"
    })
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"], errors="coerce")
    event_log = log_converter.apply(df_pm4py)
    threshold = 0.5
    parameters = {
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: threshold,
    }
    res = heuristics_miner.apply(event_log, parameters=parameters)
    if isinstance(res, tuple):
        net, im, fm = res
    else:
        net, im, fm = pt_converter.apply(res)
    start_places = set(im.keys())
    end_places = set(fm.keys())
    for i, p in enumerate(net.places):
        if p in start_places:
            p.name = "PROCESS_START"
        elif p in end_places:
            p.name = "PROCESS_END"
        else:
            p.name = f"State_{i}"
    visible_activities = []
    for j, t in enumerate(net.transitions):
        label = t.label
        if not label: # Label이 없으면 (Silent Transition / tau)
            t.name = f"ROUTING_LOGIC_{j}"
            t.label = "SILENT" 
        else:
            # 실제 업무 이름은 그대로 유지하고 목록에 추가
            t.name = label
            visible_activities.append(label)
    arcs = []
    for a in net.arcs:
        src = getattr(a.source, "name", str(a.source))
        tgt = getattr(a.target, "name", str(a.target))
        arcs.append(f"({src} -> {tgt})")
    initial_marking_str = ", ".join([f"{p.name}:{im[p]}" for p in im])
    final_marking_str   = ", ".join([f"{p.name}:{fm[p]}" for p in fm])
    lines = []
    lines.append("### PETRI NET ABSTRACTION (Process Flow)")
    lines.append(f"**Mining Parameter**: Heuristics Miner (Dependency Threshold = {threshold}).")
    lines.append("Use this structure to understand the logical sequence of activities.")
    lines.append("- **Nodes**: 'PROCESS_START' (Start), 'PROCESS_END' (End), 'State_X' (Intermediate Stages).")
    lines.append("- **Transitions**: Real activities (e.g., 'Submit') vs. 'ROUTING_LOGIC' (Invisible system logic for branching/merging).")
    lines.append("")
    lines.append("**Visible Activities (Business Steps):**")
    lines.append("[" + ", ".join(f"'{act}'" for act in visible_activities) + "]")
    lines.append("")
    lines.append("**Process Flows (Arcs):**")
    lines.append("\n".join(arcs))
    lines.append("")
    lines.append(f"**Initial State**: {initial_marking_str}")
    lines.append(f"**Final State**: {final_marking_str}")    
    return "\n".join(lines)