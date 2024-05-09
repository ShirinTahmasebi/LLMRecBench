from abc import ABC, abstractmethod
from data.item_tokens import DataTokensPool, DataTokensPoolMovieLens, DataTokensPoolAmazon
from data.item_model import RecBoleItemMovieLens, RecBoleItemAmazon


class Dataset(ABC):
    
    @abstractmethod
    def get_token_pools(self) -> DataTokensPool:
        raise NotImplementedError(f"This method should be implemented in child class for the corresponding dataset.")
    
    @classmethod
    @abstractmethod
    def get_data_tokens_pool_type(cls):
        raise NotImplementedError(f"This method should be implemented in child class for the corresponding dataset.")
    
    @classmethod
    @abstractmethod
    def get_data_item_type(cls):
        raise NotImplementedError(f"This method should be implemented in child class for the corresponding dataset.")


class DatasetMovieLens(Dataset):
    
    def __init__(self, data_tokens_pool) -> None:
        self.data_tokens_pool: DataTokensPoolMovieLens = data_tokens_pool

    def get_token_pools(self) -> DataTokensPoolMovieLens:
        return self.data_tokens_pool
    
    @classmethod
    def get_data_tokens_pool_type(cls):
        return DataTokensPoolMovieLens
    
    @classmethod
    def get_data_item_type(cls):
        return RecBoleItemMovieLens 
    
    
class DatasetAmazon(Dataset):
    
    def __init__(self, data_tokens_pool) -> None:
        self.data_tokens_pool: DataTokensPoolAmazon = data_tokens_pool
    
    def get_token_pools(self) -> DataTokensPoolAmazon:
        return self.data_tokens_pool
    
    @classmethod
    def get_data_tokens_pool_type(cls):
        return DataTokensPoolAmazon
    
    @classmethod
    def get_data_item_type(cls):
        return RecBoleItemAmazon