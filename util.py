import json
import os
import pandas as pd
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
            "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o","gpt-5.1","gpt-5.2"
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
        "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o","gpt-5.1","gpt-5.2"
    ]:

        response = model_instance.chat.completions.create(
            model= model_version,
            messages= prompt,
            temperature=0.0,
            seed = 12345,
            response_format={"type": "json_object"},
        )
        
        obj = json.loads(response.choices[0].message.content)

    return obj

def get_dfg_abstraction(df_input):
    df_pm4py = df_input[['case_id', 'timestamp', 'activity']].copy().rename(columns={
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



def get_synonym_context(df: pd.DataFrame,
                            case_col: str = 'case_id',
                            time_col: str = 'timestamp',
                            act_col: str = 'activity'):
    df_pm4py = df[[case_col, time_col, act_col]].copy()
    df_pm4py.rename(columns={
        case_col: "case:concept:name",
        time_col: "time:timestamp",
        act_col: "concept:name"
    }, inplace=True)
    df_pm4py["time:timestamp"] = pd.to_datetime(df_pm4py["time:timestamp"], errors="coerce")
    dfg, start_activities, end_activities = pm4py.discover_dfg(df_pm4py)
    def get_activity_context(activity, dfg_dict):
        predecessors = {k[0]: v for k, v in dfg_dict.items() if k[1] == activity}
        successors = {k[1]: v for k, v in dfg_dict.items() if k[0] == activity}
        total_pred = sum(predecessors.values())
        total_succ = sum(successors.values())
        def format_and_sort(dist_dict, total):
            if total == 0: return {}
            items = [(k, v/total) for k, v in dist_dict.items() if (v/total) >= 0.05]
            items.sort(key=lambda x: x[1], reverse=True)
            return {k: f"{v*100:.1f}%" for k, v in items}
        return format_and_sort(predecessors, total_pred), format_and_sort(successors, total_succ)
    all_activities = sorted(df[act_col].unique())
    flow_data_list = []
    for act in all_activities:
        pred, succ = get_activity_context(act, dfg)
        flow_data_list.append({
            'activity': act,
            'predecessors': pred,
            'successors': succ
        })
    json_flow_context = json.dumps(flow_data_list, indent=2, ensure_ascii=False)
    #json_activity_list = json.dumps(all_activities, indent=2, ensure_ascii=False)
    #vc = df_pm4py["concept:name"].value_counts().reset_index()
    #vc.columns = ["activity", "count"]
    #json_activity_counts = vc.to_json(orient="records", indent=2, force_ascii=False)
    return json_flow_context#, json_activity_list#, json_activity_counts



def get_homonym_context_deprecated(
    df_input: pd.DataFrame, 
    case_col: str = 'case_id', 
    activity_col: str = 'activity', 
    timestamp_col: str = 'timestamp',
    max_occurrence: int = 3, 
    min_support_ratio: float = 0.01
) -> str:
    df = df_input.sort_values(by=[case_col, timestamp_col]).copy()
    total_cases = df[case_col].nunique()
    min_support_count = total_cases * min_support_ratio
    df['prev_activity'] = df.groupby(case_col)[activity_col].shift(1).fillna('[START]')
    df['next_activity'] = df.groupby(case_col)[activity_col].shift(-1).fillna('[END]')
    df['occ_idx'] = df.groupby([case_col, activity_col]).cumcount() + 1
    max_occ_per_activity = df.groupby(activity_col)['occ_idx'].max()
    target_activities = max_occ_per_activity[max_occ_per_activity >= 2].index.tolist()
    df_target = df[
        (df[activity_col].isin(target_activities)) & 
        (df['occ_idx'] <= max_occurrence)
    ].copy()
    stats = defaultdict(lambda: defaultdict(lambda: {'prev': Counter(), 'next': Counter(), 'count': 0}))
    for _, row in df_target.iterrows():
        act = row[activity_col]
        nth = row['occ_idx']
        stats[act][nth]['prev'][row['prev_activity']] += 1
        stats[act][nth]['next'][row['next_activity']] += 1
        stats[act][nth]['count'] += 1
    lines = []
    lines.append("### ACTIVITY OCCURRENCE CONTEXT REPORT")
    lines.append(f"Analyzing Predecessors/Successors for up to {max_occurrence} occurrences.")
    lines.append(f"Excluding occurrences with support less than {min_support_ratio*100}% (N < {min_support_count:.0f}).\n")
    for act, occ_data in stats.items():
        act_lines = []
        for nth in sorted(occ_data.keys()):
            data = occ_data[nth]
            total = data['count']
            if total < min_support_count:
                continue
            def fmt_probs(counter):
                all_items = counter.most_common()
                return ", ".join([f"{k}({v/total*100:.1f}%)" for k, v in all_items])
            prev_str = fmt_probs(data['prev'])
            next_str = fmt_probs(data['next'])
            ordinal = {1:'1st', 2:'2nd', 3:'3rd', 4:'4th', 5:'5th'}.get(nth, f"{nth}th")
            act_lines.append(f"  - **{ordinal} Occurrence** (N={total}):")
            act_lines.append(f"    - Predecessors: [ {prev_str} ]")
            act_lines.append(f"    - Successors:   [ {next_str} ]")
        if act_lines:
            lines.append(f"## Activity: **{act}**")
            lines.extend(act_lines)
            lines.append("")
    return "\n".join(lines)


def get_homonym_context(
    df_input: pd.DataFrame, 
    case_col: str = 'case_id', 
    activity_col: str = 'activity', 
    timestamp_col: str = 'timestamp',
    min_support_ratio: float = 0.01
) -> str:
    df = df_input.sort_values(by=[case_col, timestamp_col]).copy()
    
    total_cases = df[case_col].nunique()
    min_support_count = total_cases * min_support_ratio
    
    df['prev_activity'] = df.groupby(case_col)[activity_col].shift(1).fillna('[START]')
    df['next_activity'] = df.groupby(case_col)[activity_col].shift(-1).fillna('[END]')
    
    df['total_count_in_case'] = df.groupby([case_col, activity_col])[activity_col].transform('count')
    
    max_occ_per_activity = df.groupby(activity_col)['total_count_in_case'].max()
    target_activities = max_occ_per_activity[max_occ_per_activity >= 2].index.tolist()
    
    df_target = df[df[activity_col].isin(target_activities)].copy()
    
    stats = defaultdict(lambda: defaultdict(lambda: {'prev': Counter(), 'next': Counter(), 'count': 0}))
    
    for _, row in df_target.iterrows():
        act = row[activity_col]
        count_in_case = row['total_count_in_case']
        
        if count_in_case == 1:
            group = 'Single Occurrence Case'
        else:
            group = 'Multiple Occurrence Case'
            
        stats[act][group]['prev'][row['prev_activity']] += 1
        stats[act][group]['next'][row['next_activity']] += 1
        stats[act][group]['count'] += 1
        
    lines = []
    lines.append("### SINGLE vs MULTIPLE OCCURRENCE CONTEXT REPORT")
    lines.append("Comparing contexts between 'Single Occurrence Cases' (activity appears once) and 'Multiple Occurrence Cases' (activity appears multiple times).")
    lines.append(f"Excluding groups with support less than {min_support_ratio*100}% (N < {min_support_count:.0f}).\n")
    
    for act, group_data in stats.items():
        act_lines = []
        target_groups = ['Single Occurrence Case', 'Multiple Occurrence Case']
        
        for group in target_groups:
            if group not in group_data:
                continue
                
            data = group_data[group]
            total = data['count']
            
            if total < min_support_count:
                continue
            
            def fmt_probs(counter):
                all_items = counter.most_common()
                return ", ".join([f"{k}({v/total*100:.1f}%)" for k, v in all_items])
            
            prev_str = fmt_probs(data['prev'])
            next_str = fmt_probs(data['next'])
            
            act_lines.append(f"  - **{group}** (Total Rows={total}):")
            act_lines.append(f"    - Predecessors: [ {prev_str} ]")
            act_lines.append(f"    - Successors:   [ {next_str} ]")
            
        if act_lines:
            lines.append(f"## Activity: **{act}**")
            lines.extend(act_lines)
            lines.append("")
            
    return "\n".join(lines)