from recbole.trainer import Trainer
from tqdm import tqdm
from recbole.utils import set_color
from helpers.utils_general import merge_dicts_with_matching_keys
from helpers.utils_global import *
import pandas as pd

class LLMBasedTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.number_of_candidates = config[KEYWORDS.NUMBER_OF_CANDIDATES]
        self.ground_truth_position = config[KEYWORDS.GT_POSITION] # Set to -1 if it does not matter       
        self.data_path = config[KEYWORDS.DATA_PATH]
        self.dataset_name = dataset.dataset_name
        self.output_file_name = \
            f"{self.model.get_model_name()}_" + \
            f"{self.dataset_name}_" + \
            f"history{config[KEYWORDS.NUMBER_OF_HISTORY_ITEMS]}_" + \
            f"recoms{config[KEYWORDS.NUMBER_OF_RECOMS]}_"
        
        if self.number_of_candidates > 0 and self.ground_truth_position > 0:
            self.output_file_name =  self.output_file_name + \
                f"candidates{self.number_of_candidates}_" + \
                f"gt{self.ground_truth_position}_"
            
                  
      
    def evaluate(self, eval_data, start_num=0, end_num=-1, show_progress=False):
        
        output_file_name = f"{self.output_file_name}start{start_num}_end{end_num}"
        total_items_to_be_processed = end_num - start_num if end_num > 0 else len(eval_data) * eval_data.batch_size
        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
     
        counter = 0
        result_dic_final = {}
        for index, batched_data in enumerate(iter_data):
            interaction, _, _, _ = batched_data # collate_fn in FullSortEvalDataLoader
            batch_size = len(interaction)
            first_item_in_batch_index = index * batch_size
            last_item_in_batch_index = first_item_in_batch_index + batch_size
            
            if first_item_in_batch_index < start_num:
                continue
            elif end_num > 0 and first_item_in_batch_index >= end_num:
                break
            
            log(f"Processing from {first_item_in_batch_index} to {last_item_in_batch_index} - {counter} of {total_items_to_be_processed} items processed so far!")

            result_dic = self.model.full_sort_predict(interaction)
            result_dic_final = merge_dicts_with_matching_keys(result_dic, result_dic_final)
            
            df = pd.DataFrame(result_dic_final)
            df.to_csv(f"{output_file_name}.csv", index=False)
            counter += batch_size
