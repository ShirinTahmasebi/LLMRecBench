from helpers.annotations import *

## DO NOT INSTANTIATE THIS CLASS! USE THE OBJECT IN THE GLOBAL UTILS!!

class Keywords(object):
    
    @constant
    def NUMBER_OF_CANDIDATES():
        return 'number_of_candidates' 
    
    @constant
    def NUMBER_OF_HISTORY_ITEMS():
        return 'number_of_history_items' 
    
    @constant
    def NUMBER_OF_RECOMS():
        return 'number_of_recommendations' 
    
    @constant
    def GT_POSITION():
        return 'ground_truth_position'
     
    @constant
    def DATA_PATH():
        return 'data_path' 
     
    @constant
    def DEVICE():
        return 'device'  
     
    @constant
    def USER_ID():
        return 'user_id'
     
    @constant
    def ITEM_ID():
        return 'item_id' 
     
    @constant
    def TIMESTAMP():
        return 'timestamp' 
     
    @constant
    def ITEM_ID_LIST():
        return 'item_id_list' 
     
    @constant
    def TIMESTAMP_LIST():
        return 'timestamp_list' 
     
    @constant
    def MODEL_NAME():
        return 'model_name' 
     
    @constant
    def MODEL_SHORT_NAME():
        return 'model_short_name' 
     
    @constant
    def TEMPERATURE():
        return 'temperature' 
     
    @constant
    def TOP_P():
        return 'top_p' 
     
    @constant
    def TOP_K():
        return 'top_k' 
     
    @constant
    def MAX_TOKENS():
        return 'max_tokens' 
     
    @constant
    def LORA_WEIGHTS_PATH():
        return 'lora_weights_path' 
     
    @constant
    def CHECKPOINT_MODEL_NAME():
        return 'checkpoint_model_name' 
     
    @constant
    def SHOW_PROGRESS():
        return 'show_progress' 

