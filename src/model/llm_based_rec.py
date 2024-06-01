import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type, List
from datasets import Dataset
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.interaction import Interaction as RecBoleInteraction
from data.user_interaction import UserInteractionHistory
from data.item_tokens import DataTokensPool
from helpers.utils_recbole import recbole_get_item_text
from helpers.utils_general import ModelConfig
from helpers.utils_global import *


T = TypeVar('T')
class LLMBasedRec(ABC, SequentialRecommender, Generic[T]):
    
    def __init__(self, config, dataset, load_from_checkpoint, cls: Type[T], load_model: bool):
        super().__init__(config, dataset)
        self.dataset_type_cls = cls
        self.initialize_variables(config, dataset)
        if load_model:
            self.model, self.tokenizer = self.initialize_model_tokenizer(load_from_checkpoint)
            if self.model:
                log("Model and tokenizer are loaded!")
        
        self.fake_fn = torch.nn.Linear(1, 1)
        
    
    @abstractmethod
    def initialize_model_tokenizer(self, load_from_checkpoint):
        raise NotImplementedError(f"The model initialization method is not implemented for {self.__class__.__name__}.")
    
    @abstractmethod
    def create_prompt(self):
        raise NotImplementedError(f"The prompt creation method is not implemented for {self.__class__.__name__}.")
    
    @abstractmethod
    def format_input(self, interaction: RecBoleInteraction):
        raise NotImplementedError(f"The input formatting metod is not defined for {self.__class__.__name__}.")
        
    @abstractmethod
    def inference_llm(self, model_input_txt_batch: list):
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")
           
    @abstractmethod
    def finetune_llm(self, train_dataset: Dataset, val_dataset: Dataset):
        raise NotImplementedError(f"The model finetuning is not defined for {self.__class__.__name__}.")
         
    @abstractmethod
    def process_output(self, model_output_txt_batch: list):
        raise NotImplementedError(f"The output processing metod is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def get_model_name(self):
        raise NotImplementedError(f"The model name is not defined for {self.__class__.__name__}.")
        
    @abstractmethod
    def get_train_data_hf_hub(self):
        raise NotImplementedError(f"The train data hub is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def count_tokens(self, input: str):
        raise NotImplementedError(f"The token counter is not defined for {self.__class__.__name__}.")
    
    def get_context_window_limit(self):
        return 4096
    
    def initialize_variables(self, config, dataset):
        self.number_of_recommendations = config[KEYWORDS.NUMBER_OF_RECOMS]
        self.data_path = config[KEYWORDS.DATA_PATH]
        self.dataset_name = dataset.dataset_name
        
        item_token2id = list(dataset.field2token_id[KEYWORDS.ITEM_ID].keys())
        user_token2id = list(dataset.field2token_id[KEYWORDS.USER_ID].keys())
        data_tokens_pool: DataTokensPool = recbole_get_item_text(
            data_path=self.data_path,
            dataset_name=self.dataset_name,
            user_token2id=user_token2id,
            item_token2id=item_token2id,
            token_pool_type=self.dataset_type_cls.get_data_tokens_pool_type()
        )
        
        self.dataset = self.dataset_type_cls(data_tokens_pool)
        self.model_config = ModelConfig(
            id=config[KEYWORDS.MODEL_NAME_NO_CHECKPOINT],
            model_short_name=config[KEYWORDS.MODEL_SHORT_NAME_NO_CHECKPOINT],
            api_key=ALL_API_KEYS["HF_API_KEY"],
            temperature=config[KEYWORDS.TEMPERATURE], # The range is [0, 1]
            top_p=config[KEYWORDS.TOP_P],
            top_k=config[KEYWORDS.TOP_K],
            max_tokens=config[KEYWORDS.MAX_TOKENS],
        )
        self.item_num = dataset.item_num


    def remove_hallucination(self, recommended_items_batch: list):
        # TODO: Check hallucination
        return recommended_items_batch
    
    
    def append_interations_safe_context_window(self, interactions_list: List[str]):
        dummy_prompt = self.create_prompt(" ")
        current_tokens_count = self.count_tokens([dummy_prompt])[0] + self.model_config.max_tokens
        interactions_to_be_injected = []
        
        counter = 0
        for interaction in interactions_list:
            additional_tokens_count = self.count_tokens([interaction + ", "])[0]
            
            if current_tokens_count + additional_tokens_count > self.get_context_window_limit():
                break
            
            current_tokens_count += additional_tokens_count
            interactions_to_be_injected.append(interaction)
            counter += 1
            
        final_prompt = self.create_prompt(", ".join(interactions_to_be_injected))
        return final_prompt, ", ".join(interactions_to_be_injected), counter
    
    
    def evaluate_score(self, recommended_items_batch: list, gt_names_batch: list):
        import math
        
        hit5_batch = []
        hit10_batch = []
        ndcg5_batch = []
        ndcg10_batch = []
        
        if not len(recommended_items_batch) == len(gt_names_batch):
            raise Exception("Unexpected Length Mismatch!")
        
        for i in range(len(recommended_items_batch)):
            gt = gt_names_batch[i]
            recoms = recommended_items_batch[i]
            
            if gt in recoms[:5]:
                hit5_batch.append(1)
                pos = recoms[:5].index(gt)
                ndcg5_batch.append(1.0 / (math.log(pos + 2) / math.log(2)) / 1.0)
            else:
                hit5_batch.append(0)
                ndcg5_batch.append(0)
            
            if gt in recoms[:10]:
                hit10_batch.append(1)
                pos = recoms[:10].index(gt)
                ndcg10_batch.append(1.0 / (math.log(pos + 2) / math.log(2)) / 1.0)
            else:
                hit10_batch.append(0)
                ndcg10_batch.append(0)
                
        return hit5_batch, hit10_batch, ndcg5_batch, ndcg10_batch
                
        
    def get_train_text(self, interaction: RecBoleInteraction):

        users_interactions_list = UserInteractionHistory.build(
            interaction=interaction,
            tokens=self.dataset.get_token_pools(), 
            data_item_type=self.dataset_type_cls.get_data_item_type()
        )        
        
        model_input_txt_batch, _, _ = self.format_input(users_interactions_list)
        gt_names_batch = UserInteractionHistory.get_gt_titles(users_interactions_list)

        train_data = [input + " " + response for input, response in zip(model_input_txt_batch, gt_names_batch)]
        return train_data
        
    
    def full_sort_predict(self, interaction: RecBoleInteraction):
        import time

        users_interactions_list = UserInteractionHistory.build(
            interaction=interaction,
            tokens=self.dataset.get_token_pools(), 
            data_item_type=self.dataset_type_cls.get_data_item_type()
        )
        
        user_ids_batch = UserInteractionHistory.get_user_ids(users_interactions_list)
        gt_names_batch = UserInteractionHistory.get_gt_titles(users_interactions_list)
        
        log(f"User IDs: {','.join(user_ids_batch)}")
        
        model_input_txt_batch, \
            interactions_txt_batch, \
                interactions_injected_count_batch = self.format_input(users_interactions_list)
        
        start_time = time.time_ns()
        model_output_txt_batch = self.inference_llm(model_input_txt_batch)
        end_time = time.time_ns()

        recommended_items_batch: list = self.process_output(model_output_txt_batch)        
        hit5_batch, hit10_batch, ndcg5_batch, ndcg10_batch = \
            self.evaluate_score(recommended_items_batch, gt_names_batch)
        
        no_hallu_recommended_items_batch: list = self.remove_hallucination(recommended_items_batch)
        hit5_no_hallu_batch, hit10_no_hallu_batch, ndcg5_no_hallu_batch, ndcg10_no_hallu_batch = \
            self.evaluate_score(no_hallu_recommended_items_batch, gt_names_batch)
        
        return {
            "user_id": user_ids_batch,
            "interaction_history": interactions_txt_batch,
            "number_of_interactions": interactions_injected_count_batch,
            "ground_truth": gt_names_batch,
            "recommended_items": recommended_items_batch,
            "hit@5": hit5_batch,
            "hit@10": hit10_batch, 
            "ndcg@5": ndcg5_batch, 
            "ndcg@10": ndcg10_batch,
            "no_hallu_recommended_items": no_hallu_recommended_items_batch,
            "hit@5_no_hallu": hit5_no_hallu_batch,
            "hit@10_no_hallu": hit10_no_hallu_batch, 
            "ndcg@5_no_hallu": ndcg5_no_hallu_batch, 
            "ndcg@10_no_hallu": ndcg10_no_hallu_batch,
            "average_execution_time_ns": [(end_time - start_time) // len(user_ids_batch)] * len(user_ids_batch),
        }
    