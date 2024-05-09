from model.llm_based_rec import LLMBasedRec
from helpers.utils_llm import import_hf_model_and_tokenizer, import_genrec_model_and_tokenizer
from helpers.utils_general import last_non_zero_index, log, get_absolute_path
from data.user_interaction import UserInteractionHistory, RecBoleItemMovieLens
from typing import List
from prompts.prompts_genrec import *


class GenRec(LLMBasedRec):
    
    def __init__(self, config, dataset, model_config, load_from_checkpoint=True):
        self.number_of_history_items = 10
        self.lora_weights_path = config['lora_weights_path']
        # TODO: Checkpoint path should be in config file. But, the checkpoint name should be chosen by the dataset type.
        self.checkpoint_model_name = config['checkpoint_model_name']
        super().__init__(config, dataset, model_config, load_from_checkpoint)

 
    def initialize_model_tokenizer(self, load_from_checkpoint):
        if load_from_checkpoint:
            model, tokenizer = import_genrec_model_and_tokenizer(
                model_name=self.checkpoint_model_name, 
                access_token=self.model_config.api_key,
                lora_weights=get_absolute_path(self.lora_weights_path)
            )
        else:
            model, tokenizer = import_hf_model_and_tokenizer(
                model_name=self.model_config.id, 
                access_token=self.model_config.api_key
            )        
        
        return model, tokenizer

    
    def get_model_name(self):
        return "GenRec"
    
    
    def create_prompt(self, input):
        formatted_interactions = MOVIE_INTERACTIONS.format(input=input)
        return f"""{MOVIE_INSTRUCTION} \n\n {formatted_interactions} \n\n {MOVIE_OUTPUT}"""


    def format_input(self, user_history_batch: List[UserInteractionHistory]):
        interactions_prompt_txt_batch = []
        interactions_txt_batch = []
        
        # TODO: Can the type be generic so that the boilerplate code be removed?
        for history_per_user in user_history_batch:
            if type(history_per_user.ground_truth_item) is RecBoleItemMovieLens:
                his_items_count = last_non_zero_index(RecBoleItemMovieLens.get_item_ids(history_per_user.interaction_items)) + 1
                start_index = his_items_count - self.number_of_history_items
                end_index = his_items_count 
                
                zipped_list = zip(
                    RecBoleItemMovieLens.get_item_titles(history_per_user.interaction_items)[start_index:end_index],
                    RecBoleItemMovieLens.get_item_years(history_per_user.interaction_items)[start_index:end_index]
                )
                
                output_interaction = [f"{title} ({year})" for title, year in zipped_list]
            interactions_prompt_txt_batch.append(self.create_prompt(", ".join(output_interaction)))    
            interactions_txt_batch.append(", ".join(output_interaction))
                
        return interactions_prompt_txt_batch, interactions_txt_batch

        
    def call_llm(self, model_input_txt_batch: list):
        all_results = []
        for input in model_input_txt_batch:
            input_ids = self.tokenizer(input, return_tensors='pt').input_ids.cuda()
            
            result = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=self.model_config.max_tokens,
                do_sample=True, 
                top_p=0.01, 
                temperature=self.model_config.temperature,
            )
            
            text_result = self.tokenizer.batch_decode(
                result.detach().cpu().numpy(), 
                skip_special_tokens=True
            )[0]
            
            log(f"RESULT: {text_result} \n\n")

            all_results.append(text_result.split(MOVIE_OUTPUT)[1])
        
        log("\n".join(all_results))
        return all_results

        
    def process_output(self, model_output_txt_batch: list):
        processed_batch = []
        for txt in model_output_txt_batch:
            processed_batch.append(txt)
        return processed_batch  
    
    