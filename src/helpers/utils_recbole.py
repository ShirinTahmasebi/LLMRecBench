from data.item_tokens import DataTokensPool

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
    

def recbole_get_item_text(
    data_path: str, 
    dataset_name: str, 
    user_token2id: list, 
    item_token2id: list, 
    token_pool_type
) -> DataTokensPool:
    import os.path as osp
    feat_path = osp.join(data_path, f'{dataset_name}.item')    
    return token_pool_type(feat_path=feat_path, user_token2id=user_token2id, item_token2id=item_token2id)
  