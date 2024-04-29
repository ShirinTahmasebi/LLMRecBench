from recbole.trainer import Trainer
from tqdm import tqdm
from recbole.utils import set_color
from utils import recbole_get_item_text, last_non_zero_index, merge_dicts_with_matching_keys
import pandas as pd

class LLMBasedTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.number_of_candidates = config['number_of_candidates']
        self.ground_truth_position = config['ground_truth_position'] # Set to -1 if it does not matter
        # TODO: Use this variable!
        self.max_history_len = config['max_history_len']
        
        self.item_token2id = list(dataset.field2token_id['item_id'].keys())
        
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        
        # TODO: THis part is movielens specific. Revise it!
        self.item_text, self.item_year, self.item_genre = recbole_get_item_text(
            data_path=self.data_path,
            dataset_name=self.dataset_name,
            item_token2id=self.item_token2id
        )
                  
      
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
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
        for _, batched_data in enumerate(iter_data):
            interaction, _, _, _ = batched_data # collate_fn in FullSortEvalDataLoader
            result_dic = self.model.full_sort_predict(interaction)
            result_dic_final = merge_dicts_with_matching_keys(result_dic, result_dic_final)
            
            df = pd.DataFrame(result_dic_final)
            # TODO: Choose a more informative name for the results
            df.to_csv(f"{self.dataset_name}_{self.model.get_model_name()}.csv", index=False)
            counter += len(interaction)
            
            if counter > 10:
                break
               
                    
    def print_interactions(self, eval_data, show_progress=False):
        from datetime import datetime
        
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
        
        
        for _, batched_data in enumerate(iter_data):
            interaction, _, _, _ = batched_data
            
            for i in range(len(interaction)):
                user_id = interaction[i]['user_id']
                his_item_ids = interaction[i]['item_id_list']
                timestamp_list = interaction[i]['timestamp_list']
                gt_id =  self.item_token2id[interaction[i]['item_id']]
                gt_timestamp = interaction[i]['timestamp']
                gt_title = self.item_text[interaction[i]['item_id']]
                
                history_ids = []
                history_titles = []
                history_timestampts = []
                his_items_count = last_non_zero_index(his_item_ids) + 1
                start_index = his_items_count - 10 # Prints the last 10 history items
                end_index = his_items_count   
                            
                for j, idx in enumerate(his_item_ids[start_index:end_index]):
                    movie_id = self.item_token2id[idx]
                    movie_title = self.item_text[idx]
                    movie_year = self.item_year[idx]
                    timestamp = timestamp_list[j]
                    
                    history_ids.append(movie_id)
                    history_titles.append(movie_title)
                    history_timestampts.append(str(datetime.fromtimestamp(timestamp.item())))
                
                user_output = f"""
                    User: {user_id} \n
                    GT ID: {gt_id} \n
                    GT Title: {gt_title} \n
                    GT Time: {datetime.fromtimestamp(gt_timestamp.item())} \n
                    History IDs:  {", ".join(history_ids)} \n
                    History Titles:  {", ".join(history_titles)} \n
                    History Time:  {", ".join(history_timestampts)} \n
                    ---------------------------------
                    """
                print(user_output)
