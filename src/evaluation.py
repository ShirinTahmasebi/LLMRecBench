
import re
import csv
import pandas as pd
from helpers.utils_general import get_absolute_path
from helpers.utils_global import log
    
def calculate_metrics(file_names):
    output = {}
    results = {}
    
    for file_name in file_names:
        match = re.search(r"(.*)\_history(\d+)\_recoms.*\.csv", file_name)
        if match:
            model_name = match.group(1)
            num = match.group(2)
            key = f"history_{num}"
            
            if model_name not in output:
                results = {}
                
            if key not in results:
                results[key] = {
                    "hit@5": 0, "ndcg@5": 0, "hit@10": 0, "ndcg@10": 0,
                    "hallucination_percentage": 0, "repetition_percentage": 0
                }

            df = pd.read_csv(get_absolute_path(file_name))            
            results[key]["average_number_of_candidate"] = df["number_of_interactions"].mean()
            results[key]["hit@5"] = df['hit@5'].mean() * 100
            results[key]["ndcg@5"] = df['ndcg@5'].mean() * 100
            results[key]["hit@10"] = df['hit@10'].mean() * 100
            results[key]["ndcg@10"] = df['ndcg@10'].mean() * 100
            results[key]["hallucination_percentage"] = 5  # Example fixed value
            results[key]["repetition_percentage"] = 5     # Example fixed value
            output[model_name] = results
    return output

if __name__=="__main__":
    results = calculate_metrics([
        "GenRec_amazon-toys-games_history10_recoms10_start0_end600.csv",
        "GenRec_amazon-toys-games_history15_recoms10_start0_end600.csv",
        "GenRec_amazon-toys-games_history20_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history10_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history20_recoms10_start0_end600_2.csv",
        "GenRec_ml-1m_history12_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history12_recoms10_start0_end600_2.csv",
        "GenRec_ml-1m_history15_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history15_recoms10_start0_end600_2.csv",
        "GenRec_ml-1m_history17_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history20_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history25_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history30_recoms10_start0_end600.csv",
        "GenRec_ml-1m_history35_recoms10_start0_end600.csv",
    ])
    
    log(results)
    
    pass