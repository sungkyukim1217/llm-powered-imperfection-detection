import pandas as pd
from collections import defaultdict
import pm4py
from pm4py.objects.log.util import dataframe_utils
from util import *

def analyze_and_visualize_log(log_name):
    print(f"\n{'='*20} Start analyzing [{log_name}] log {'='*20}")
    
    file_route = f"./dataset/{log_name}.csv"
    
    # 1. Extract the list of Injected Activities by directly reading the CSV file
    df_csv = pd.read_csv(file_route)
    
    act_col = 'Activity' if 'Activity' in df_csv.columns else 'activity'
    
    if 'label' in df_csv.columns:
        injected_activities = set(df_csv[df_csv['label'].notna()][act_col].unique())
    else:
        injected_activities = set()
        
    # 2. Load and convert the log
    df_raw, _ = build_event_jsons(log_name=file_route, chunk_cases=1)
    
    df_log = df_raw[['case_id', 'timestamp', 'activity']].copy().rename(columns={
        'case_id': "case:concept:name",
        'timestamp': "time:timestamp",
        'activity': "concept:name"
    })
    
    df_log["time:timestamp"] = pd.to_datetime(df_log["time:timestamp"], errors="coerce")
    df_log = dataframe_utils.convert_timestamp_columns_in_df(df_log)
    event_log = log_converter.apply(df_log)

    heu_net = pm4py.discover_heuristics_net(event_log)
    image_path = f"{log_name}_heuristics.png"
    pm4py.save_vis_heuristics_net(heu_net, image_path)
    print(f"Image saved successfully: {image_path}")

    incoming_arcs = defaultdict(list)
    outgoing_arcs = defaultdict(list)

    for (source, target), weight in heu_net.dfg.items():
        outgoing_arcs[source].append(target)
        incoming_arcs[target].append(source)

    # 3. Categorize and sort activities
    activities = heu_net.nodes.keys()
    
    injected_list = []
    clean_list = []
    
    for act in activities:
        in_count = len(incoming_arcs.get(act, []))
        if act in injected_activities:
            injected_list.append((act, in_count))
        else:
            clean_list.append((act, in_count))
            
    # Sort in descending order based on incoming count
    injected_list = sorted(injected_list, key=lambda x: x[1], reverse=True)
    clean_list = sorted(clean_list, key=lambda x: x[1], reverse=True)

    # 4. Save to txt file
    txt_path = f"{log_name}_incoming_counts.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== [{log_name}] Number of incoming arcs per activity after noise filtering ===\n\n")

        # Write [Injected Activities] section
        f.write("=== [Injected Activities] ===\n")
        if not injected_list:
            f.write("  None\n\n")
        for act, count in injected_list:
            f.write(f"▶ Activity: {act}\n")
            f.write(f"  <- Incoming count: {count}\n")
            f.write("-" * 50 + "\n")
            
        # Write [Clean Activities] section
        f.write("\n=== [Clean Activities] ===\n")
        if not clean_list:
            f.write("  None\n\n")
        for act, count in clean_list:
            f.write(f"▶ Activity: {act}\n")
            f.write(f"  <- Incoming count: {count}\n")
            f.write("-" * 50 + "\n")

    print(f"Text file saved successfully: {txt_path}")
        
    return heu_net, incoming_arcs, outgoing_arcs, image_path

imperfect = 'credit_seed1_03_homonymous'
#imperfect = 'pub_seed1_03_homonymous'

heu_imperfect, in_imperfect, out_imperfect, path_imperfect = analyze_and_visualize_log(imperfect)