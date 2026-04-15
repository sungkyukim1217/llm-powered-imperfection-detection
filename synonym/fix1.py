def run_fix1(df, mapping_dict):
    """
    Applies the identified synonym mappings to the event log to normalize activity labels.
    
    This function performs the actual data correction (repair) phase. It first flattens the 
    'mapping_dict' into a lookup table where each synonym variant points to its designated 
    canonical (clean) label. Then, it iterates through the event log and replaces all 
    occurrences of variant labels with their standardized counterparts. Activities not 
    present in the mapping remain unchanged. This process ensures consistent naming 
    conventions across the entire dataset, which is critical for accurate process discovery 
    and conformance checking.

    Args:
        df: The pandas DataFrame representing the event log to be corrected.
        mapping_dict: A dictionary where keys are canonical labels and values are lists of 
                      their synonymous variants (output from run_step4).

    Returns:
        pd.DataFrame: A new DataFrame with normalized activity labels in the 'activity' column.
    """
    print(">>> Starting Activity Correction...")
    
    synonym_to_clean = {}
    for clean_label, variants in mapping_dict.items():
        for var in variants:
            synonym_to_clean[var] = clean_label
            
    df_fixed = df.copy()
    df_fixed['activity'] = df_fixed['activity'].map(synonym_to_clean).fillna(df_fixed['activity'])
    
    print(f">>> Correction complete. Total rows processed: {len(df_fixed)}")
    return df_fixed

