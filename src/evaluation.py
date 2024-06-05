from helpers.utils_general import get_absolute_path
from helpers.utils_evaluation import calculate_metrics, check_filters_and_return_info, preprocess_results
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
    
    result["meta_stats"] = dict(sorted(meta_stats.items()))
    
    
    chart_info = {}
    import re
    pattern = re.compile(r"GenRec_ml-1m_history(\d+)_recoms10_temp([\d\w]+)")

    for key, value in result["meta_stats"].items():
        match = pattern.match(key)
        if match:
            history = int(match.group(1))
            temp = f"temp{match.group(2)}"
            hit5 = value["average_hit5"]
            ndcg5 = value["average_ndcg5"]
            hit10 = value["average_hit10"]
            ndcg10 = value["average_ndcg10"]
            
            if temp not in chart_info:
                chart_info[temp] = {"history": [], "hit@5": [], "ndcg@5": [], "hit@10": [], "ndcg@10": []}
            
            chart_info[temp]["history"].append(history)
            chart_info[temp]["hit@5"].append(hit5)
            chart_info[temp]["ndcg@5"].append(ndcg5)
            chart_info[temp]["hit@10"].append(hit10)
            chart_info[temp]["ndcg@10"].append(ndcg10)

    result["chart_info"] = chart_info
    
    print(result)
    return result

def plot_chart(key, value, name_prefix):
    import matplotlib.pyplot as plt

    history = value["history"]
    plt.figure(figsize=(10, 6))

    for metric in ["hit@5", "ndcg@5", "hit@10", "ndcg@10"]:
        plt.plot(history, value[metric], label=metric)
    
    plt.xlabel("History")
    plt.ylabel("Values")
    plt.title(f"Metrics for {key}")
    plt.legend()
    plt.grid(True)
    
    plots_path = get_absolute_path("results/plots")
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    filename = os.path.join(plots_path, f"{name_prefix} - {key}.png")
    plt.savefig(filename)
    plt.close()
    

if __name__ == '__main__':
    model_name = "GenRec"
    dataset_name = "ml-1m"
    preprocess_results(get_absolute_path("results"))
    results = execute(
        base_path=get_absolute_path("results_processed"), 
        filters={
            "model_name": [model_name],
            "dataset_name": [dataset_name],
            "recoms": [10],
            "history": [10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50],
            "temp": [0.6, 1],
        }
    )
    
    log(dict(sorted(results.items())))
    
    chart_info = results["chart_info"]
    for temperature_key, value in chart_info.items():
        plot_chart(temperature_key, value, name_prefix=f"Chart - {model_name} - {dataset_name}")
    log("Done!")
