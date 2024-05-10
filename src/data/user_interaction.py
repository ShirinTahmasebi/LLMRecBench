import dataclasses 
from typing import List
from datetime import datetime
from recbole.data.interaction import Interaction as RecBoleInteraction
from helpers.utils_general import reshape_tensor_remove_zero_from_end
from data.item_model import DataItemModel
from data.item_tokens import DataTokensPool
   
@dataclasses.dataclass
class UserInteractionHistory: 

    user_id: str
    interaction_items: List[DataItemModel]
    ground_truth_item: DataItemModel
    
    @classmethod
    def get_user_ids(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.user_id, interactions_list))
    
    @classmethod
    def get_gt_ids(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.ground_truth_item.item_id, interactions_list))
    
    @classmethod
    def get_gt_titles(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.ground_truth_item.item_title, interactions_list))

    @classmethod
    def build(cls, interaction: RecBoleInteraction, tokens: DataTokensPool, data_item_type):
        user_interaction_list = []
        
        for i in range(len(interaction)):
            user_id = tokens.user_token2id[interaction[i]['user_id']]
            his_item_ids = reshape_tensor_remove_zero_from_end(interaction[i]['item_id_list'])
            timestamp_list = interaction[i]['timestamp_list']
            ground_truth = data_item_type.build_ground_truth(tokens, interaction, i)
            
            interaction_items = []       
            for j, idx in enumerate(his_item_ids):
                interaction_item = data_item_type.build_interaction(tokens, timestamp_list, idx, j)
                interaction_items.append(interaction_item)
                
            user_interaction_item = UserInteractionHistory(
                user_id=user_id,
                interaction_items=interaction_items,
                ground_truth_item=ground_truth
            )
            
            user_interaction_list.append(user_interaction_item)
        return user_interaction_list
    
    
    def __str__(self):
        history_txt = '\n'.join(f'Item #{i}: {str(item)}' for i, item in enumerate(self.interaction_items))
        return f"""
    User: {self.user_id} \n
    GT ID: {self.ground_truth_item.item_id} \n
    GT Title: {self.ground_truth_item.item_title} \n
    GT Time: {datetime.fromtimestamp(self.ground_truth_item.item_timestamp)} \n
    History Items: \n {history_txt} \n
    ---------------------------------
    """
