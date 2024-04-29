import dataclasses


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
    

def get_absolute_path(path_relative_to_project_root):
    import os
    current_directory = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    final_directory = os.path.join(
        current_directory,
        rf'../../{path_relative_to_project_root}'
    )
    return final_directory


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