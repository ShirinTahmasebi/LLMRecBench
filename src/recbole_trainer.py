from recbole.trainer import Trainer
from tqdm import tqdm
from recbole.utils import set_color
from helpers.utils_general import merge_dicts_with_matching_keys, huggingface_api_login
from helpers.utils_global import *
import pandas as pd

class LLMBasedTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.number_of_candidates = config[KEYWORDS.NUMBER_OF_CANDIDATES]
        self.number_of_users_to_train = config[KEYWORDS.NUMBER_OF_USERS_TO_TRAIN]
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


    def load_or_create_train_dataset(self, train_data):
        huggingface_api_login(ALL_API_KEYS["HF_HUB_KEY"])
        
        data_text = []
        set_of_user_ids = set()
        list_of_user_ids = []
        list_of_interactions_count = []
        
        for _, interactions_batch in enumerate(train_data):
            if len(set_of_user_ids) >= self.number_of_users_to_train:
                log(f"""
                    ---------------------------------------- 
                    Training Data Details:
                        Number of Unique Users = {len(set_of_user_ids)}
                        Number of Records = {len(list_of_user_ids)}
                        Number of Interactions per All Users = {set(list_of_interactions_count)}
                    ----------------------------------------
                    """)
                break
            user_ids_list = interactions_batch['user_id'].tolist()
            number_of_interactions = [len(per_user) for per_user in interactions_batch['item_id_list']]
            list_of_user_ids.extend(user_ids_list)
            list_of_interactions_count.extend(number_of_interactions)
            set_of_user_ids.update(user_ids_list)
            text_batch = self.model.get_train_text(interactions_batch)
            data_text.extend(text_batch)
        
        jsonl_file_path = get_absolute_path(f'{self.model.get_model_name()}_train.jsonl')

        with open(jsonl_file_path, 'a') as outfile:
            for item in data_text:
                outfile.write('{"text": "' + item.replace('\n', '\\n') + '"}')
                outfile.write('\n')
        
        from datasets import load_dataset
        dataset = load_dataset('json', data_files=jsonl_file_path, split='train')
        dataset.push_to_hub(self.model.get_train_data_hf_hub())
        
        return dataset
    
    
    def train(self, train_data, valid_data, start_num=0, end_num=-1, show_progress=False):
        try:
            from datasets import load_dataset
            dataset = load_dataset(self.model.get_train_data_hf_hub(), use_auth_token=True)
        except Exception as e:
            dataset = self.load_or_create_train_dataset(train_data)
