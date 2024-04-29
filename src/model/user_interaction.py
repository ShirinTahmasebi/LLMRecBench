from recbole.data.interaction import Interaction as RecBoleInteraction
from helpers.utils_general import last_non_zero_index
import dataclasses 
import datetime
from typing import List


@dataclasses.dataclass
class UserInteraction:

    user_id: str
    history_ids: list
    history_titles: list
    history_years: list
    history_timestampts: list
    gt_id: str
    gt_title: str
    gt_timestamp: str
    
    @classmethod
    def get_user_ids(cls, interactions_list: List['UserInteraction']):
        return list(map(lambda obj: obj.user_id, interactions_list))
    
    @classmethod
    def get_gt_ids(cls, interactions_list: List['UserInteraction']):
        return list(map(lambda obj: obj.gt_id, interactions_list))
    
    @classmethod
    def get_gt_titles(cls, interactions_list: List['UserInteraction']):
        return list(map(lambda obj: obj.gt_title, interactions_list))

    @classmethod
    def build(cls, interaction: RecBoleInteraction, item_token2id, item_text, item_year):
        user_interaction_list = []
        
        for i in range(len(interaction)):
            user_id = interaction[i]['user_id']
            his_item_ids = interaction[i]['item_id_list']
            timestamp_list = interaction[i]['timestamp_list']
            gt_id =  item_token2id[interaction[i]['item_id']]
            gt_title = item_text[interaction[i]['item_id']]
            gt_timestamp = interaction[i]['timestamp'].item()
           
            history_ids = []
            history_titles = []
            history_years = []
            history_timestampts = []
                         
            for j, idx in enumerate(his_item_ids):
                movie_id = item_token2id[idx]
                movie_title = item_text[idx]
                movie_year = item_year[idx]
                timestamp = timestamp_list[j].item()
                
                history_ids.append(movie_id)
                history_titles.append(movie_title)
                history_years.append(movie_year)
                history_timestampts.append(timestamp)
            
            user_interaction_item = UserInteraction(
                user_id=user_id,
                history_ids=history_ids,
                history_titles=history_titles,
                history_years=history_years,
                history_timestampts=history_timestampts,
                gt_id=gt_id,
                gt_title=gt_title,
                gt_timestamp=gt_timestamp
            )
            
            user_interaction_list.append(user_interaction_item)
        
        return user_interaction_list
    
    
    def __str__(self):
        return f"""
    User: {self.user_id} \n
    GT ID: {self.gt_id} \n
    GT Title: {self.gt_title} \n
    GT Time: {datetime.fromtimestamp(self.gt_timestamp)} \n
    History IDs:  {", ".join(self.history_ids)} \n
    History Titles:  {", ".join(self.history_titles)} \n
    History Time:  {", ".join(self.history_timestampts)} \n
    ---------------------------------
    """