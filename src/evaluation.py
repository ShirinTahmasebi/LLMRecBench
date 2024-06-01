from helpers.utils_general import get_absolute_path
from helpers.utils_evaluation import calculate_metrics, check_filters_and_return_info
from helpers.utils_global import *
import pandas as pd
import os


def execute(base_path, filters):
    result = {}
    meta = {}
    
    for root, _, files in os.walk(base_path):
        for file in files:
            log(f"File Name: {file}")
            if file.endswith(".csv"):
                
                info = check_filters_and_return_info(filters, root, file)
                if not info:
                    continue

                file_id = info["file_id"]
                model_name = info["model_name"]
                dataset_name = info["dataset_name"]
                recoms = info["recoms"]
                history = info["history"]
                temp = info["temp"]
                start = info["start"]
                end = info["end"]
                
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                metrics = calculate_metrics(df)
                
                entry = {
                    "id": file_id,
                    "file_name": file,
                    "recoms": recoms,
                    "history": history,
                    "temp": temp,
                    "start": start,
                    "end": end,
                    **metrics
                }
                                
                if model_name not in result:
                    result[model_name] = {}
                if dataset_name not in result[model_name]:
                    result[model_name][dataset_name] = []  
                result[model_name][dataset_name].append(entry)
                
                meta_key = f"{model_name}_{dataset_name}_history{history}_recoms{recoms}_temp{temp.replace('.', 'p')}"  
                if meta_key not in meta:
                    meta[meta_key] = []
                meta[meta_key].append(entry)
                
    meta_stats = {}
    for key, list_of_dicts in meta.items():
        total_hit5 = total_ndcg5 = total_hit10 = total_ndcg10 = 0
        count = len(list_of_dicts)
        
        for d in list_of_dicts:
            total_hit5 += d["hit@5"]
            total_ndcg5 += d["ndcg@5"]
            total_hit10 += d["hit@10"]
            total_ndcg10 += d["ndcg@10"]
        
        average_hit5 = total_hit5 / count if count > 0 else 0
        average_ndcg5 = total_ndcg5 / count if count > 0 else 0
        average_hit10 = total_hit10 / count if count > 0 else 0
        average_ndcg10 = total_ndcg10 / count if count > 0 else 0
        
        meta_stats[key] = {
            "average_hit5": average_hit5, 
            "average_ndcg5": average_ndcg5,
            "average_hit10": average_hit10, 
            "average_ndcg10": average_ndcg10,
        }
    
    result["meta_stats"] = meta_stats
    
    return result


if __name__ == '__main__':
    
    results = execute(
        base_path=get_absolute_path("results_processed"), 
        filters={
            "model_name": ["GenRec"],
            "dataset_name": ["ml-1m"],
            "recoms": [10],
            "history": [10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50],
            "temp": [.6, 1],
        }
    )
    
    log(results)
    log("Done!")
