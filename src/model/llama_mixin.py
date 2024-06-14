from data.user_interaction import UserInteractionHistory
from helpers.utils_general import last_non_zero_index
from typing import List
from data.dataset import DatasetMovieLens, DatasetAmazon
from prompts.prompts_genrec import *

class LLaMaMixin:
    
    def create_prompt(self, input):
        if self.dataset_type_cls is DatasetMovieLens:
            instruction = MOVIE_INSTRUCTION
        if self.dataset_type_cls is DatasetAmazon:
            instruction = TOYS_INSTRUCTION
        
        formatted_interactions = INTERACTIONS.format(input=input)
        return f"""{instruction}

{formatted_interactions}
    
{OUTPUT}"""

    
    def format_input(self, user_history_batch: List[UserInteractionHistory]):
        interactions_prompt_txt_batch = []
        interactions_txt_batch = []
        interactions_injected_count_batch = []
        
        for history_per_user in user_history_batch:
            data_item_cls = self.dataset.get_data_item_type()
            his_items_count = last_non_zero_index(data_item_cls.get_item_ids(history_per_user.interaction_items)) + 1
            if self.number_of_history_items > his_items_count:
                start_index = 0
                end_index = his_items_count
            else:
                start_index = his_items_count - self.number_of_history_items
                end_index = his_items_count 
            output_interaction = data_item_cls.get_interactions_text(
                history_per_user.interaction_items,
                start_index,
                end_index
            )
            
            final_prompt, valid_interactions_txt, injected_interactions_count = self.append_interations_safe_context_window(output_interaction)
            interactions_prompt_txt_batch.append(final_prompt)
            interactions_txt_batch.append(valid_interactions_txt)
            interactions_injected_count_batch.append(injected_interactions_count)
                
        return interactions_prompt_txt_batch, interactions_txt_batch, interactions_injected_count_batch
