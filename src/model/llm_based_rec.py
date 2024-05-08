from recbole.model.abstract_recommender import SequentialRecommender
from recbole.data.interaction import Interaction as RecBoleInteraction
from model.user_interaction import UserInteraction
import torch
from helpers.utils_recbole import recbole_get_item_text
from helpers.utils_general import log
from abc import ABC, abstractmethod

class LLMBasedRec(ABC, SequentialRecommender):
    
    def __init__(self, config, dataset, model_config, load_from_checkpoint):
        super().__init__(config, dataset)
        self.initialize_variables(config, dataset, model_config)
        self.model, self.tokenizer = self.initialize_model_tokenizer(load_from_checkpoint)
        self.fake_fn = torch.nn.Linear(1, 1)
        if self.model:
            log("Model and tokenizer are loaded!")
        
    
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
    def call_llm(self, model_input_txt_batch: list):
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")
         
    @abstractmethod
    def process_output(self, model_output_txt_batch: list):
        raise NotImplementedError(f"The output processing metod is not defined for {self.__class__.__name__}.")
    
    @abstractmethod
    def get_model_name(self):
        raise NotImplementedError(f"The model name is not defined for {self.__class__.__name__}.")
    
    def initialize_variables(self, config, dataset, model_config):
        self.item_token2id = list(dataset.field2token_id['item_id'].keys())
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        # TODO: THis part is movielens specific. Revise it!
        self.item_text, self.item_year, self.item_genre = recbole_get_item_text(
            data_path=self.data_path,
            dataset_name=self.dataset_name,
            item_token2id=self.item_token2id
        )
        self.model_config = model_config
        self.item_num = dataset.item_num
        self.number_of_recommendations = config['number_of_recommendations']


    def remove_hallucination(self, recommended_items_batch: list):
        # TODO: Check hallucination
        return recommended_items_batch
        
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
                
        
        
    def full_sort_predict(self, interaction: RecBoleInteraction):
        users_interactions_list = UserInteraction.build(
            interaction=interaction,
            item_token2id=self.item_token2id,
            item_text=self.item_text,
            item_year=self.item_year,
        )
        
        user_ids_batch = UserInteraction.get_user_ids(users_interactions_list)
        gt_names_batch = UserInteraction.get_gt_titles(users_interactions_list)
        
        model_input_txt_batch, interactions_txt_batch = self.format_input(users_interactions_list)
        model_output_txt_batch = self.call_llm(model_input_txt_batch)
        
        recommended_items_batch: list = self.process_output(model_output_txt_batch)        
        hit5_batch, hit10_batch, ndcg5_batch, ndcg10_batch = \
            self.evaluate_score(recommended_items_batch, gt_names_batch)
        
        no_hallu_recommended_items_batch: list = self.remove_hallucination(recommended_items_batch)
        hit5_no_hallu_batch, hit10_no_hallu_batch, ndcg5_no_hallu_batch, ndcg10_no_hallu_batch = \
            self.evaluate_score(no_hallu_recommended_items_batch, gt_names_batch)
        
        return {
            "user_id": user_ids_batch,
            "interaction_history": interactions_txt_batch,
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
            "ndcg@10_no_hallu": ndcg10_no_hallu_batch
        }
    