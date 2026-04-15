def run_fix1(df, mapping_dict):

    """
    Standardizes activity labels by replacing detected polluted variants with their canonical clean roots.
    
    This function acts as the final repair module in the polluted label detection pipeline. 
    It constructs a reverse lookup table from the 'mapping_dict', where each polluted variant 
    (e.g., 'Task_001', 'Task_ID99') is mapped to its corresponding 'Clean Label'. 
    Using this map, the function performs a vectorized replacement across the event log's 
    activity column. Labels that were not identified as polluted are preserved in their 
    original form. This normalization is essential for reducing event log complexity and 
    merging identical process steps that were artificially split by mutable identifiers.

    Args:
        df: The pandas DataFrame representing the event log to be repaired.
        mapping_dict: A dictionary mapping canonical clean labels to their respective 
                      lists of polluted variants (output from run_step3).

    Returns:
        pd.DataFrame: A corrected DataFrame with normalized activity names, ensuring 
                      consistent labeling for subsequent process mining analysis.
    """
    print(">>> Starting Activity Correction...")
    polluted_to_clean = {}
    for clean_label, variants in mapping_dict.items():
        for var in variants:
            polluted_to_clean[var] = clean_label
    df_fixed = df.copy()
    df_fixed['activity'] = df_fixed['activity'].map(polluted_to_clean).fillna(df_fixed['activity'])
    
    print(f">>> Correction complete. Total rows processed: {len(df_fixed)}")
    return df_fixed

