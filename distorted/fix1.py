def run_fix1(df, mapping_dict):

    """
    Cleanses the event log by replacing identified typographical distortions with their canonical clean labels.
    
    This function executes the final repair action for the distorted label pipeline. It flattens 
    the 'mapping_dict' into a high-performance lookup table where each detected typo (variant) 
    points to its linguistically correct or frequency-dominant 'Clean Label'. By mapping 
    these variants back to their targets, the function eliminates redundant activity nodes 
    caused by human error or system noise. This process significantly improves the clarity 
    of discovered process models and ensures that frequency-based metrics are not diluted 
    by character-level variations.

    Args:
        df: The pandas DataFrame representing the event log to be cleansed.
        mapping_dict: A dictionary where keys are canonical clean labels and values are 
                      lists of their identified distorted variants (output from run_step2).

    Returns:
        pd.DataFrame: A corrected DataFrame with unified activity names in the 'activity' column.
    """
    print(">>> Starting Activity Correction...")
    
    distorted_to_clean = {}
    for clean_label, variants in mapping_dict.items():
        for var in variants:
            distorted_to_clean[var] = clean_label
            
    # Copy dataframe and apply mapping
    df_fixed = df.copy()
    df_fixed['activity'] = df_fixed['activity'].map(distorted_to_clean).fillna(df_fixed['activity'])
    
    print(f">>> Correction complete. Total rows processed: {len(df_fixed)}")
    return df_fixed
