import dataclasses 
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod
from helpers.utils_global import *

        
@dataclasses.dataclass
class DataItemModel(ABC):
    item_id: str
    item_title: str
    item_timestamp: str
    
    def __str__(self):
        return f"""
            ID: {self.item_id} - 
            Title: {self.item_title} - 
            Timestamp: {datetime.fromtimestamp(self.item_timestamp)}
        """
    
    @classmethod
    def get_item_ids(cls, interaction_items: List['DataItemModel']):
        return list(map(lambda obj: obj.item_id, interaction_items))
    
    @classmethod
    def get_item_titles(cls, interaction_items: List['DataItemModel']):
        return list(map(lambda obj: obj.item_title, interaction_items))
    
    @classmethod
    @abstractmethod
    def get_interactions_text(cls, interaction_items, start_index, end_index):
        raise NotImplementedError(f"This method should be implemented in child class for the corresponding dataset.")



@dataclasses.dataclass
class DataItemModelMovieLens(DataItemModel):
    item_year: str
    item_genre: str
    
    @classmethod
    def get_item_years(cls, interaction_items: List['DataItemModel']):
        return list(map(lambda obj: obj.item_year, interaction_items))
    
    @classmethod
    def build_ground_truth(cls, tokens, interaction, i):
        gt_id =  tokens.item_token2id[interaction[i][KEYWORDS.ITEM_ID]]
        gt_title = tokens.item_token2text[interaction[i][KEYWORDS.ITEM_ID]]
        gt_year = tokens.item_token2release_year[interaction[i][KEYWORDS.ITEM_ID]]
        gt_genre = tokens.item_token2genre[interaction[i][KEYWORDS.ITEM_ID]]
        gt_timestamp = interaction[i][KEYWORDS.TIMESTAMP].item()
        
        return DataItemModelMovieLens(
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
        
        return DataItemModelMovieLens(
            item_id=movie_id,
            item_title=movie_title,
            item_timestamp=movie_timestamp,
            item_year=movie_year,
            item_genre=movie_genre,
        )
    
    @classmethod
    def get_interactions_text(cls, interaction_items, start_index, end_index):
        zipped_list = zip(
            cls.get_item_titles(interaction_items)[start_index:end_index],
            cls.get_item_years(interaction_items)[start_index:end_index]
        )
        output_interaction = [f"{title} ({year})" for title, year in zipped_list]
        return output_interaction

   
@dataclasses.dataclass
class DataItemModelAmazon(DataItemModel):
    
    @classmethod
    def build_ground_truth(cls, tokens, interaction, i):
        gt_id =  tokens.item_token2id[interaction[i][KEYWORDS.ITEM_ID]]
        gt_title = tokens.item_token2text[interaction[i][KEYWORDS.ITEM_ID]]
        gt_timestamp = interaction[i][KEYWORDS.TIMESTAMP].item()
        
        return DataItemModelAmazon(
            item_id=gt_id,
            item_title=gt_title,
            item_timestamp=gt_timestamp
        )
        
    @classmethod
    def build_interaction(cls, tokens, timestamp_list, idx, j):
        item_id = tokens.item_token2id[idx]
        item_title = tokens.item_token2text[idx]
        item_timestamp = timestamp_list[j].item()
        
        return DataItemModelAmazon(
            item_id=item_id,
            item_title=item_title,
            item_timestamp=item_timestamp,
        )
    
    @classmethod
    def get_interactions_text(cls, interaction_items, start_index, end_index):
        return cls.get_item_titles(interaction_items)[start_index:end_index]
