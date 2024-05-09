from abc import ABC, abstractmethod
from helpers.annotations import singleton
import dataclasses 

# https://github.com/RUCAIBox/LLMRank/blob/b3999d8fdacc0ad1520d7b036b28589f0e7e6479/llmrank/utils.py#L14
def get_model(model_name):
    from recbole.utils import get_model as recbole_get_model
    import importlib

    if importlib.util.find_spec(f'model.{model_name.lower()}', __name__):
        model_module = importlib.import_module(f'model.{model_name.lower()}', __name__)
        model_class = getattr(model_module, model_name)
        return model_class
    else:
        return recbole_get_model(model_name)
    

@dataclasses.dataclass
class RecBoleItemTokens(ABC):
    item_token2id: list
    item_token2text: list
    
    def __new__(cls, *args, **kwargs):
        if cls is RecBoleItemTokens:
            raise TypeError(f"{cls.__name__} class may not be instantiated directly")
        return super().__new__(cls)


@singleton
class RecBoleItemTokensMovieLens(RecBoleItemTokens):
    item_token2release_year: list
    item_token2genre: list
    
    # https://github.com/RUCAIBox/LLMRank/blob/b3999d8fdacc0ad1520d7b036b28589f0e7e6479/llmrank/utils.py
    def __init__(self, feat_path: str, item_token2id: list):
        token_text = {}
        token_release_year = {}
        token_genre = {}
        item_token2text = ['[PAD]']
        item_token2release_year = [0]
        item_token2genre = ['[PAD]']
    
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                item_id, movie_title, release_year, genre = line.strip().split('\t')
                token_text[item_id] = movie_title
                token_release_year[item_id] = release_year
                token_genre[item_id] = genre
                
        for i, token in enumerate(item_token2id):
            if token == '[PAD]': continue
            raw_text = token_text[token]
            if raw_text.endswith(', The'):
                raw_text = 'The ' + raw_text[:-5]
            elif raw_text.endswith(', A'):
                raw_text = 'A ' + raw_text[:-3]
            item_token2text.append(raw_text)
            item_token2release_year.append(token_release_year[token])
            item_token2genre.append(token_genre[token])
            
  
        super().__init__(item_token2id, item_token2text)
        self.item_token2release_year = item_token2release_year
        self.item_token2genre = item_token2genre
        
        


@singleton
class RecBoleItemTokensAmazon(RecBoleItemTokens):
        
    # https://github.com/RUCAIBox/LLMRank/blob/b3999d8fdacc0ad1520d7b036b28589f0e7e6479/llmrank/utils.py    
    def __init__(self, feat_path: str, item_token2id: list):
        token_text = {}
        item_token2text = ['[PAD]']
    
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                item_id, title = line.strip().split('\t')
                token_text[item_id] = title
        
        for i, token in enumerate(item_token2id):
            if token == '[PAD]': continue
            raw_text = token_text[token]
            item_token2text.append(raw_text)
            
        super().__init__(item_token2id, item_token2text)
        


def recbole_get_item_text(data_path: str, dataset_name: str, item_token2id: list) -> RecBoleItemTokens:
    import os.path as osp
    feat_path = osp.join(data_path, f'{dataset_name}.item')
    
    if dataset_name in ['ml-1m', 'ml-1m-full']:
        return RecBoleItemTokensMovieLens(feat_path=feat_path, item_token2id=item_token2id)
    elif dataset_name in ['Games', 'Games-6k']:
        return RecBoleItemTokensAmazon(feat_path=feat_path, item_token2id=item_token2id)
    else:
        raise NotImplementedError()
    
  