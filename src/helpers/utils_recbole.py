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
    
    
def recbole_get_item_text(data_path: str, dataset_name: str, item_token2id: list):
    # Returns a dict for items. The key in this dict is item ids and the value is the item name.
    # It only works with the title. Not genre, summary, year, or etc.
    # For working with other fields, we can modify this function to make the dict more detailed and informative.
    token_text = {}
    token_release_year = {}
    token_genre = {}
    item_text = ['[PAD]']
    item_release_year = [0]
    item_genre = ['[PAD]']
    import os.path as osp
    feat_path = osp.join(data_path, f'{dataset_name}.item')
    
    if dataset_name in ['ml-1m', 'ml-1m-full']:
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
            item_text.append(raw_text)
            item_release_year.append(token_release_year[token])
            item_genre.append(token_genre[token])
        return item_text, item_release_year, item_genre
    elif dataset_name in ['Games', 'Games-6k']:
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                item_id, title = line.strip().split('\t')
                token_text[item_id] = title
        for i, token in enumerate(item_token2id):
            if token == '[PAD]': continue
            raw_text = token_text[token]
            item_text.append(raw_text)
        return item_text
    else:
        raise NotImplementedError()
    
  