import dataclasses

@dataclasses.dataclass
class ModelConfig:
    id: str
    model_short_name: str
    api_key: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    

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
    

def reshape_tensor_remove_zero_from_end(tensor):
    import numpy as np
    arr = np.array(tensor)
    zero_index = np.argmax(arr == 0)
    if zero_index > 0:
        sliced_array = tensor[:zero_index]
        return sliced_array
    return tensor


def huggingface_api_login(api_token):
    from huggingface_hub import HfApi, HfFolder
    HfFolder.save_token(api_token)
    api = HfApi()
    

# TODO: I think utils_llm.py is a better place for having this function.    
from transformers import TrainerCallback
class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps, output_dir):
        self.save_steps = save_steps
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        from helpers.utils_global import log
        if state.global_step % self.save_steps == 0:
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            kwargs["model"].save_pretrained(checkpoint_dir)
            kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            log(f"Saved checkpoint at step {state.global_step} to {checkpoint_dir}")
