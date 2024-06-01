import re
import os
from helpers.utils_global import log

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


def parse_result_filename(filename):
    pattern = r"temp(?P<temp>\w+)_start(?P<start>\d+)_end(?P<end>\d+)(\_\d+)*\.csv"
    import re
    match = re.match(pattern, filename)
    if match:
        return match.group('temp'), \
            match.group('temp').replace("p", "."), \
            int(match.group('start')), \
            int(match.group('end'))
    return None, None, None, None


def parse_result_directory(directory_name):
    pattern = r"history(?P<history>\d+)_recoms(?P<recoms>\d+)"
    match = re.match(pattern, directory_name)
    if match:
        return int(match.group('history')), int(match.group('recoms'))
    return None, None


def process_single_result_file(input_file, output_file):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        lines = infile.readlines()
        
        pattern = re.compile(r'^\d{4},')
        
        current_line = ''
        
        for line in lines:
            if pattern.match(line):
                if current_line:
                    outfile.write(current_line + '\n')
                current_line = line.strip()
            else:
                current_line += ' ' + line.strip()
        
        if current_line:
            outfile.write(current_line + '\n')


def preprocess_results(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            log(f"File Name: {file}")
            file_path = os.path.join(root, file)
            pattern = r"temp(?P<temp>\w+)_start(?P<start>\d+)_end(?P<end>\d+)(\_\d+)*\.csv"
            match = re.match(pattern, file)
            
            if match:
                output_path = file_path.replace("results", "results_processed")
                process_single_result_file(input_file=file_path, output_file=output_path)


def check_filters_and_return_info(filters, root, file):
    file_path = os.path.join(root, file)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    history, recoms = parse_result_directory(parent_dir)
    _, temp, start, end = parse_result_filename(file)
    
    if history is None or recoms is None or temp is None or start is None or end is None:
        return None
    
    file_name = os.path.basename(file_path)
    model_name = grandparent_dir.split("_")[0]
    dataset_name = grandparent_dir.split("_")[1]
    file_id = f"{grandparent_dir}_{parent_dir}_{file_name}"
    
    output_dict = {
        "file_id": file_id, 
        "model_name": model_name, 
        "dataset_name": dataset_name, 
        "recoms": recoms, 
        "history": history, 
        "temp": temp, 
        "start": start, 
        "end": end
    }
    
    if (model_name not in filters["model_name"] or
        dataset_name not in filters["dataset_name"] or
        int(recoms) not in filters["recoms"] or
        int(history) not in filters["history"] or
        float(temp) not in filters["temp"]):
            return None
                
    return output_dict
