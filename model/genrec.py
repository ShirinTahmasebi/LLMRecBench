from recbole.data.interaction import Interaction as RecBoleInteraction
from model.llm_based_rec import LLMBasedRec
from datetime import datetime
from utils import import_hf_model_and_tokenizer, last_non_zero_index, LLAMA_PROMPT_FORMAT

class GenRec(LLMBasedRec):
    
    def __init__(self, config, dataset, model_config):
        super().__init__(config, dataset, model_config)
        self.number_of_history_items = 10

 
    def initialize_model_tokenizer(self):
        return import_hf_model_and_tokenizer(
            model_name=self.model_config.id, 
            access_token=self.model_config.api_key    
        )

    
    def get_model_name(self):
        return "GenRec"
    
    
    def create_prompt(self, input):
        instruction_txt = """Instruction: Given the movie viewing habits, what is the most probable movie they will choose to watch next?"""
        input_txt = f"""Input: {input}"""
        output_txt = "Output:"
        
        return f"""{instruction_txt} \n\n {input_txt} \n\n {output_txt}"""


    def format_input(self, interaction: RecBoleInteraction):
        interactions_prompt_txt_batch = []
        interactions_txt_batch = []
        ground_truth_id_batch = []
        ground_truth_name_batch = []
        
        for i in range(len(interaction)):
            user_id = interaction[i]['user_id']
            his_item_ids = interaction[i]['item_id_list']
            gt_id =  self.item_token2id[interaction[i]['item_id']]
            gt_title = self.item_text[interaction[i]['item_id']]
            
            history_ids = []
            history_titles = []
            output_interaction = []
            his_items_count = last_non_zero_index(his_item_ids) + 1
            start_index = his_items_count - self.number_of_history_items
            end_index = his_items_count   
                         
            for _, idx in enumerate(his_item_ids[start_index:end_index]):
                movie_id = self.item_token2id[idx]
                movie_title = self.item_text[idx]
                movie_year = self.item_year[idx]
                
                history_ids.append(movie_id)
                history_titles.append(movie_title)
                output_interaction.append(f"{movie_title} ({movie_year})")
                
            interactions_prompt_txt_batch.append(self.create_prompt(", ".join(output_interaction)))    
            interactions_txt_batch.append(", ".join(output_interaction))
            ground_truth_id_batch.append(gt_id)    
            ground_truth_name_batch.append(gt_title)    
                
        return interactions_prompt_txt_batch, interactions_txt_batch, ground_truth_id_batch, ground_truth_name_batch


    def format_input_demo(self, interaction: RecBoleInteraction, gt_ids: list, candidate_items: list):
        output = []
        for i in range(len(interaction)):
            user_id = interaction[i]['user_id']
            his_item_ids = interaction[i]['item_id_list']
            timestamp_list = interaction[i]['timestamp_list']
            gt_id =  self.item_token2id[interaction[i]['item_id']]
            gt_title = self.item_text[interaction[i]['item_id']]
            gt_timestamp = interaction[i]['timestamp']
            
            history_ids = []
            history_titles = []
            history_timestampts = []
            his_items_count = last_non_zero_index(his_item_ids) + 1
            start_index = his_items_count-self.number_of_history_items
            end_index = his_items_count
            for j, idx in enumerate(his_item_ids[start_index:end_index]):
                movie_id = self.item_token2id[idx]
                movie_title = self.item_text[idx]
                timestamp = timestamp_list[j]
                
                history_ids.append(movie_id)
                history_titles.append(movie_title)
                history_timestampts.append(str(datetime.fromtimestamp(timestamp.item())))
                
            user_output = f"""
                User: {user_id} \n
                GT ID: {gt_id} \n
                GT Title: {gt_title} \n
                GT Time: {datetime.fromtimestamp(gt_timestamp.item())} \n
                History IDs:  {", ".join(history_ids)} \n
                History Titles:  {", ".join(history_titles)} \n
                History Time:  {", ".join(history_timestampts)} \n
                ---------------------------------
                """
            
            
            
            output.append(user_output)
                    
        return output
    
            
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

            all_results.append(text_result.split("Output: ")[1])
        
        return all_results

        
    def process_output(self, model_output_txt_batch: list):
        processed_batch = []
        for txt in model_output_txt_batch:
            processed_batch.append(txt)
        return processed_batch  
    
    