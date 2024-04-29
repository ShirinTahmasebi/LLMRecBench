import dataclasses
import importlib

def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(
        current_directory,
        rf'./{path_relative_to_project_root}'
    )

    return final_directory

# https://github.com/RUCAIBox/LLMRank/blob/b3999d8fdacc0ad1520d7b036b28589f0e7e6479/llmrank/utils.py#L14
def get_model(model_name):
    from recbole.utils import get_model as recbole_get_model
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
    
    
def import_hf_model_and_tokenizer(model_name: str, access_token: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login
    import torch
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    login(token=access_token)
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=True,
        device_map="auto",
        token=access_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return language_model, tokenizer


LLAMA_PROMPT_FORMAT = ("""<s>[INST] <<SYS>>{system_prompt}\n<</SYS>>\n{prompt} [/INST]\n""")


# GPT_CHAT_PROMPT_FORMAT = ChatPromptTemplate.from_messages(
#     [
#         ("system", "{system_prompt}"),
#         ("human", "{prompt}"),
#     ]
# )

@dataclasses.dataclass
class ModelConfig:
    id: str
    model_short_name: str
    prompt_format: str
    api_key: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
 
@dataclasses.dataclass
class TaskConfig:
    system_prompt: str
    max_length: int = 100


def last_non_zero_index(tensor):
    for i in range(len(tensor) - 1, -1, -1):
        if tensor[i] != 0:
            return i
    # If all elements are zero, return 0
    return 0


def merge_dicts_with_matching_keys(dict1, dict2):
    if not dict1:
        return dict2
    
    if not dict2:
        return dict1
    
    merged_dict = {}
    for key in dict1.keys():
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
    return merged_dict


def merge_dicts_with_non_matching_keys(dict1, dict2):
    merged_dict = {}
    for key in dict1.keys():
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        else:
            merged_dict[key] = dict1[key]
    for key in dict2.keys():
        if key not in dict1:
            merged_dict[key] = dict2[key]
    return merged_dict