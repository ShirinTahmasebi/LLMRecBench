from recbole.data.interaction import Interaction as RecBoleInteraction
import dataclasses 
import datetime
from typing import TypeVar, Generic, List
from helpers.utils_general import reshape_tensor_remove_zero_from_end
from helpers.utils_recbole import RecBoleItemTokens, RecBoleItemTokensMovieLens, RecBoleItemTokensAmazon
from enum import Enum

        
@dataclasses.dataclass
class RecBoleItem:
    item_id: str
    item_title: str
    item_timestamp: str
    
    @classmethod
    def get_item_ids(cls, interaction_items: List['RecBoleItem']):
        return list(map(lambda obj: obj.item_id, interaction_items))
    
    @classmethod
    def get_item_titles(cls, interaction_items: List['RecBoleItem']):
        return list(map(lambda obj: obj.item_title, interaction_items))


@dataclasses.dataclass
class RecBoleItemMovieLens(RecBoleItem):
    item_year: str
    item_genre: str
    
    @classmethod
    def get_item_years(cls, interaction_items: List['RecBoleItem']):
        return list(map(lambda obj: obj.item_year, interaction_items))
    
    @classmethod
    def build_ground_truth(cls, tokens, interaction, i):
        gt_id =  tokens.item_token2id[interaction[i]['item_id']]
        gt_title = tokens.item_token2text[interaction[i]['item_id']]
        gt_year = tokens.item_token2release_year[interaction[i]['item_id']]
        gt_genre = tokens.item_token2genre[interaction[i]['item_id']]
        gt_timestamp = interaction[i]['timestamp'].item()
        
        return RecBoleItemMovieLens(
            item_id=gt_id,
            item_title=gt_title,
            item_timestamp=gt_timestamp,
            item_year=gt_year,
            item_genre=gt_genre,
        )
        
    @classmethod
    def build_interaction(cls, tokens, timestamp_list, idx, j):
        movie_id = tokens.item_token2id[idx]
        movie_title = tokens.item_token2text[idx]
        movie_year = tokens.item_token2release_year[idx]
        movie_genre = tokens.item_token2genre[idx]
        movie_timestamp = timestamp_list[j].item()
        
        return RecBoleItemMovieLens(
            item_id=movie_id,
            item_title=movie_title,
            item_timestamp=movie_timestamp,
            item_year=movie_year,
            item_genre=movie_genre,
        )

   
@dataclasses.dataclass
class RecBoleItemAmazon(RecBoleItem):
    item_category: str

   
# T = TypeVar('T', bound=RecBoleItem)
@dataclasses.dataclass
class UserInteractionHistory: #(Generic[T]):

    user_id: str
    interaction_items: List[RecBoleItem] #List[T]
    ground_truth_item: RecBoleItem #T

    
    @classmethod
    def get_user_ids(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.user_id, interactions_list))
    
    # TODO: The following two functions should be moved to RecBoleItem class.
    @classmethod
    def get_gt_ids(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.ground_truth_item.item_id, interactions_list))
    
    @classmethod
    def get_gt_titles(cls, interactions_list: List['UserInteractionHistory']):
        return list(map(lambda obj: obj.ground_truth_item.item_title, interactions_list))

    @classmethod
    def build(cls, interaction: RecBoleInteraction, tokens: RecBoleItemTokens):
        user_interaction_list = []
        
        for i in range(len(interaction)):
            user_id = interaction[i]['user_id'].item()
            his_item_ids = reshape_tensor_remove_zero_from_end(interaction[i]['item_id_list'])
            timestamp_list = interaction[i]['timestamp_list']
            
            if type(tokens) is type(RecBoleItemTokensMovieLens()):
                ground_truth = RecBoleItemMovieLens.build_ground_truth(tokens, interaction, i)
            elif type(tokens) is type(RecBoleItemTokensAmazon()):
                # TODO: Fill this!
                pass
            
            interaction_items = []
                         
            for j, idx in enumerate(his_item_ids):
                if type(tokens) is type(RecBoleItemTokensMovieLens()):
                    interaction_item = RecBoleItemMovieLens.build_interaction(tokens, timestamp_list, idx, j)
                    interaction_items.append(interaction_item)
                elif type(tokens) is type(RecBoleItemTokensAmazon()):
                    pass

            
            user_interaction_item = UserInteractionHistory(
                user_id=user_id,
                interaction_items=interaction_items,
                ground_truth_item=ground_truth
            )
            
            user_interaction_list.append(user_interaction_item)
        return user_interaction_list
    
    
    # def __str__(self):
    #     return f"""
    # User: {self.user_id} \n
    # GT ID: {self.gt_id} \n
    # GT Title: {self.gt_title} \n
    # GT Time: {datetime.fromtimestamp(self.gt_timestamp)} \n
    # History IDs:  {", ".join(self.history_ids)} \n
    # History Titles:  {", ".join(self.history_titles)} \n
    # History Time:  {", ".join(self.history_timestampts)} \n
    # ---------------------------------
    # """
    
class DatasetNameEnum(Enum):
    MOVIE_LENS = 'ml-1m'
    AMAZON_TOY_GAMES = 'amazon-toys-games'

    @staticmethod
    def get_dataset_name(name: str):
        try:
            return DatasetNameEnum[name].value
        except KeyError:
            raise ValueError(f"This name is not supported as a dataset: {name}")


# TODO: it is not used!
        
class RecBoleItemTypeEnum(Enum):
    MOVIE_LENS = RecBoleItemMovieLens
    AMAZON_TOY_GAMES = RecBoleItemAmazon

    @staticmethod
    def get_class_by_name(name: str):
        try:
            return RecBoleItemTypeEnum[name].value
        except KeyError:
            raise ValueError(f"No interaction item class found for name: {name}")
        
