import os
import re
import pandas as pd
from helpers.utils_general import get_absolute_path
from helpers.utils_global import log
from decimal import Decimal

def calculate_metrics(df):
    return {
        "average_number_of_candidate": df["number_of_interactions"].mean(),
        "hit@5": df['hit@5'].mean() * 100,
        "ndcg@5": df['ndcg@5'].mean() * 100,
        "hit@10": df['hit@10'].mean() * 100,
        "ndcg@10": df['ndcg@10'].mean() * 100,
        "hallucination_percentage": 0,
        "repetition_percentage": 0,
    }

def parse_filename(filename):
    pattern = r"temp(?P<temp>\w+)_start(?P<start>\d+)_end(?P<end>\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        return match.group('temp').replace("p", "."), int(match.group('start')), int(match.group('end'))
    return None, None, None

def parse_directory(directory_name):
    pattern = r"history(?P<history>\d+)_recoms(?P<recoms>\d+)"
    match = re.match(pattern, directory_name)
    if match:
        return int(match.group('history')), int(match.group('recoms'))
    return None, None

def check_filters(filters, model_name, dataset_name, recoms, history, temp):
    if model_name not in filters["model_name"]:
        return True
    
    if dataset_name not in filters["dataset_name"]:
        return True
    
    if int(recoms) not in filters["recoms"]:
        return True
    
    if int(history) not in filters["history"]:
        return True
    
    if float(temp) not in filters["temp"]:
        return True
    
    return False
    
def main(base_path, filters):
    result = {}
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                history, recoms = parse_directory(parent_dir)
                temp, start, end = parse_filename(file)
                
                if history is None or recoms is None or temp is None or start is None or end is None:
                    continue
                
                file_name = os.path.basename(file_path)
                model_name = grandparent_dir.split("_")[0]
                dataset_name = grandparent_dir.split("_")[1]
                
                should_skip = check_filters(filters, model_name, dataset_name, recoms, history, temp)
                
                if should_skip:
                    continue

                df = pd.read_csv(file_path)
                metrics = calculate_metrics(df)
                file_id = f"{grandparent_dir}_{parent_dir}_{file_name}"
                
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
    
    return result


base_path = get_absolute_path("results")
result = main(
    base_path, 
    filters={
        "model_name": ["GenRec"],
        "dataset_name": ["ml-1m"],
        "recoms": [10],
        "history": [35, 40, 45],
        "temp": [0.6, 1],
    }
)
log(result)