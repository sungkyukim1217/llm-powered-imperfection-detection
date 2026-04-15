def run_step4(clusters, activity_counts):
    """
    Finalizes the synonym detection by selecting a representative label for each cluster based on frequency.
    
    This function performs the final mapping of synonymous groups into a 'Clean Label' and its 
    associated 'Variants'. For each cluster containing two or more activities, it identifies the 
    most frequent activity (based on 'activity_counts') and designates it as the canonical or 
    'clean' label. All other activities in the cluster are categorized as synonymous variants 
    to be normalized. This ensures that the process model remains consistent by favoring the 
    most established business terminology used in the event log.

    Args:
        clusters: A list of lists, where each sub-list contains semantically equivalent activities.
        activity_counts: A dictionary mapping each activity label to its total occurrence count 
                         in the event log, used as a proxy for naming authority.

    Returns:
        dict: A prediction dictionary where keys are the 'Clean Labels' (most frequent) 
              and values are lists of their corresponding 'Synonym Variants'.
    """
    
    print(f">>> Running Step 4  ...")

    prediction = {}
    for cluster in clusters:
        if len(cluster) < 2: continue
        clean_label = max(cluster, key=lambda x: activity_counts.get(x, 0))
        variants = sorted([label for label in cluster if label != clean_label])
        prediction[clean_label] = variants
    return prediction