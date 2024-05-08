from recbole.trainer import Trainer
from tqdm import tqdm
from recbole.utils import set_color
from helpers.utils_recbole import recbole_get_item_text
from helpers.utils_general import last_non_zero_index, merge_dicts_with_matching_keys
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
            
            # TODO: Remove this line if it is not in the debugging mode
            if counter > 10:
                break
