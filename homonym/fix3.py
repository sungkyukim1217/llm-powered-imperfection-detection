import pandas as pd
import numpy as np

def run_fix3(current_df, res_f2):
    print("- Starting Fix3: Mapping predictions for evaluation")
    new_df = current_df.copy()
    id_to_restored = {}
    for target_label, event_list in res_f2.items():
        for event in event_list:
            eid = str(event['event_id'])
            id_to_restored[eid] = event['restored_activity']
    new_df['restored_activity'] = np.nan
    new_df['restored_activity'] = new_df['event_id'].astype(str).map(id_to_restored)
    return new_df