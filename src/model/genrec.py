from typing import List, TypeVar, Type
from transformers import GenerationConfig
from prompts.prompts_genrec import *
from model.llm_based_rec import LLMBasedRec
from data.user_interaction import UserInteractionHistory
from data.dataset import DatasetMovieLens, DatasetAmazon
from helpers.utils_llm import import_hf_model_and_tokenizer, import_genrec_model_and_tokenizer
from helpers.utils_general import last_non_zero_index, get_absolute_path
from helpers.utils_global import *
import torch

T = TypeVar('T')
class GenRec(LLMBasedRec[T]):
    
    def __init__(self, config, dataset, load_from_checkpoint=True, cls: Type[T]= None):
        self.number_of_history_items = config[KEYWORDS.NUMBER_OF_HISTORY_ITEMS]
        self.lora_weights_path = config[KEYWORDS.LORA_WEIGHTS_PATH]
        self.checkpoint_model_name = config[KEYWORDS.CHECKPOINT_MODEL_NAME]
        super().__init__(config, dataset, load_from_checkpoint, cls)
        
    def finetune_llm(self, train_dataset, val_dataset):
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")
      
    def initialize_model_tokenizer(self, load_from_checkpoint):
        if self.dataset_type_cls is DatasetMovieLens:
            lora_weights_path_full = self.lora_weights_path + "/movies"
        elif self.dataset_type_cls is DatasetAmazon:
            lora_weights_path_full = self.lora_weights_path + "/toys"

        if load_from_checkpoint:
            model, tokenizer = import_genrec_model_and_tokenizer(
                model_name=self.checkpoint_model_name, 
                access_token=self.model_config.api_key,
                lora_weights=get_absolute_path(lora_weights_path_full)
            )
        else:
            model, tokenizer = import_hf_model_and_tokenizer(
                model_name=self.model_config.id, 
                access_token=self.model_config.api_key
            )        
        
        return model, tokenizer
    

    def get_model_name(self):
        return "GenRec"
    
    def get_train_data_hf_hub(self):
        return ALL_API_KEYS["HF_DATASET_REPO_NAME"]
    
    def count_tokens(self, inputs_list: list):
        tokens_list = [self.tokenizer(input, return_tensors='pt').input_ids[0] for input in inputs_list]
        len_list = [len(tokens) for tokens in tokens_list]
        return len_list
    
    def create_prompt(self, input):
        if self.dataset_type_cls is DatasetMovieLens:
            instruction = MOVIE_INSTRUCTION
        if self.dataset_type_cls is DatasetAmazon:
            instruction = TOYS_INSTRUCTION
        
        formatted_interactions = INTERACTIONS.format(input=input)
        return f"""{instruction} \n\n {formatted_interactions} \n\n {OUTPUT}"""


    def format_input(self, user_history_batch: List[UserInteractionHistory]):
        interactions_prompt_txt_batch = []
        interactions_txt_batch = []
        interactions_injected_count_batch = []
        
        for history_per_user in user_history_batch:
            data_item_cls = self.dataset.get_data_item_type()
            his_items_count = last_non_zero_index(data_item_cls.get_item_ids(history_per_user.interaction_items)) + 1
            if self.number_of_history_items > his_items_count:
                start_index = 0
                end_index = his_items_count
            else:
                start_index = his_items_count - self.number_of_history_items
                end_index = his_items_count 
            output_interaction = data_item_cls.get_interactions_text(
                history_per_user.interaction_items,
                start_index,
                end_index
            )
            
            final_prompt, valid_interactions_txt, injected_interactions_count = self.append_interations_safe_context_window(output_interaction)
            interactions_prompt_txt_batch.append(final_prompt)
            interactions_txt_batch.append(valid_interactions_txt)
            interactions_injected_count_batch.append(injected_interactions_count)
                
        return interactions_prompt_txt_batch, interactions_txt_batch, interactions_injected_count_batch

        
    def inference_llm(self, model_input_txt_batch: list):
        all_results = []
        for input in model_input_txt_batch:
            torch.cuda.empty_cache()
            input_ids = self.tokenizer(input, return_tensors='pt').input_ids.cuda()
            log(f"Input Length: {len(input_ids[0])} - {self.count_tokens([input])}")
            
            # https://github.com/huggingface/transformers/blob/4fdf58afb72b0754da30037fc800b6044e7d9c99/src/transformers/generation/configuration_utils.py#L62
            generation_config = GenerationConfig(
                max_new_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                num_return_sequences=self.number_of_recommendations, 
                do_sample=True
            )
            
            with torch.no_grad():
                results = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config
                )
            
                text_results = self.tokenizer.batch_decode(
                    results.detach().cpu().numpy(), 
                    skip_special_tokens=True
                )
                        
            recommendations = [s.split(OUTPUT)[1] for s in text_results if s.count(',') >= 1]

            log(f"Result: {recommendations} \n\n")

            all_results.append(";;".join(recommendations))
        
        return all_results

        
    def process_output(self, model_output_txt_batch: list):
        processed_batch = []
        for txt in model_output_txt_batch:
            processed_batch.append(txt)
        return processed_batch  
    
    