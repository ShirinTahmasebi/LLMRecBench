from datasets import Dataset
from prompts.prompts_genrec import *
from model.llm_based_rec import LLMBasedRec
from data.dataset import DatasetMovieLens, DatasetAmazon
from helpers.utils_llm import import_hf_model_and_tokenizer
from helpers.utils_general import get_absolute_path, SaveCheckpointCallback
from helpers.utils_global import *
from typing import TypeVar, Type
from recbole.data.interaction import Interaction as RecBoleInteraction



T = TypeVar('T')
class LLMRec(LLMBasedRec[T]):
    
    def __init__(self, config, dataset, load_from_checkpoint=False, cls: Type[T]= None, load_model=True):
        self.number_of_history_items = config[KEYWORDS.NUMBER_OF_HISTORY_ITEMS]
        self.checkpoint_path_parent = config[KEYWORDS.CHECKPOINT_PATH]
        
        self.training_args = {
            KEYWORDS.FINETUNING_LORA_ALPHA: config[KEYWORDS.FINETUNING_LORA_ALPHA],
            KEYWORDS.FINETUNING_LORA_DROPOUT: config[KEYWORDS.FINETUNING_LORA_DROPOUT],
            KEYWORDS.FINETUNING_LORA_R: config[KEYWORDS.FINETUNING_LORA_R],
            KEYWORDS.FINETUNING_TRAIN_ARGS_TRAIN_BATCH_SIZE: config[KEYWORDS.FINETUNING_TRAIN_ARGS_TRAIN_BATCH_SIZE],
            KEYWORDS.FINETUNING_TRAIN_ARGS_EVAL_BATCH_SIZE: config[KEYWORDS.FINETUNING_TRAIN_ARGS_EVAL_BATCH_SIZE],
            KEYWORDS.FINETUNING_TRAIN_ARGS_LEARNING_RATE: config[KEYWORDS.FINETUNING_TRAIN_ARGS_LEARNING_RATE],
            KEYWORDS.FINETUNING_TRAIN_ARGS_LOGGING_STEPS: config[KEYWORDS.FINETUNING_TRAIN_ARGS_LOGGING_STEPS],
            KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_STEPS: config[KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_STEPS],
            KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_SEQ_LEN: config[KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_SEQ_LEN],
        }
        
        super().__init__(config, dataset, load_from_checkpoint, cls, load_model)
       
       
    def initialize_model_tokenizer(self, load_from_checkpoint): 
        if self.dataset_type_cls is DatasetMovieLens:
            self.checkpoint_path = self.checkpoint_path_parent + "/movies"
        elif self.dataset_type_cls is DatasetAmazon:
            self.checkpoint_path = self.checkpoint_path_parent + "/toys"
            
        if load_from_checkpoint:
            # TODO: Fill later and load the fine-tuned models.
            # TODO: Use ckeckpoint_path here!
            model, tokenizer = "", ""
        else:
            model, tokenizer = import_hf_model_and_tokenizer(
                model_name=self.model_config.id, 
                access_token=self.model_config.api_key
            )        
        
        return model, tokenizer
    
    
    def create_prompt(self):
        raise NotImplementedError(f"The prompt creation method is not implemented for {self.__class__.__name__}.")
    
    def format_input(self, interaction: RecBoleInteraction):
        raise NotImplementedError(f"The input formatting metod is not defined for {self.__class__.__name__}.")
        
    def inference_llm(self, model_input_txt_batch: list):
        raise NotImplementedError(f"The model text generation method is not defined for {self.__class__.__name__}.")
         
    def process_output(self, model_output_txt_batch: list):
        raise NotImplementedError(f"The output processing metod is not defined for {self.__class__.__name__}.")
    
    def get_model_name(self):
        return "LLMRec"        
    
    def get_train_data_hf_hub(self):
        return ALL_API_KEYS["HF_DATASET_REPO_NAME"]
        
    def count_tokens(self, inputs_list: str):
        tokens_list = [self.tokenizer(input, return_tensors='pt').input_ids[0] for input in inputs_list]
        len_list = [len(tokens) for tokens in tokens_list]
        return len_list

    def finetune_llm(self, train_dataset: Dataset, val_dataset: Dataset):
        import datetime
        import wandb
        from transformers import TrainingArguments
        from peft import LoraConfig
        from trl import SFTTrainer
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project=ALL_API_KEYS["WANDB_PROJECT_NAME"],  
            entity=ALL_API_KEYS["WANDB_USERNAME"],
            name=f"{self.get_model_name()} - {current_time}"
        )
        
        log(f"""
            ---------------------
            Data Stats: 
                Train Size = {len(train_dataset)}
                Val Size = {len(val_dataset)}
                Steps per Epoch = {len(train_dataset) // self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_TRAIN_BATCH_SIZE]}
                Batch Size (Training) = {self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_TRAIN_BATCH_SIZE]}
                Batch Size (Val) = {self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_EVAL_BATCH_SIZE]}
                Max Steps = {self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_STEPS]}
                Learning Rate = {self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_LEARNING_RATE]}
            ---------------------
            """)
        
        output_dir = get_absolute_path(self.checkpoint_path)

        peft_config = LoraConfig(
            lora_alpha=self.training_args[KEYWORDS.FINETUNING_LORA_ALPHA],
            lora_dropout=self.training_args[KEYWORDS.FINETUNING_LORA_DROPOUT],
            r=self.training_args[KEYWORDS.FINETUNING_LORA_R],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        save_checkpoint_callback = SaveCheckpointCallback(save_steps=15, output_dir=output_dir)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_TRAIN_BATCH_SIZE],
            per_device_eval_batch_size=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_EVAL_BATCH_SIZE],
            gradient_accumulation_steps=4,
            learning_rate=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_LEARNING_RATE],
            logging_steps=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_LOGGING_STEPS],
            max_steps=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_STEPS],
            fp16=True, # Use mixed precision training to improve memory usage and computation speed. 
            report_to="wandb",
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.training_args[KEYWORDS.FINETUNING_TRAIN_ARGS_MAX_SEQ_LEN],
            tokenizer=self.tokenizer,
            args=training_args,
            callbacks=[save_checkpoint_callback]
        )
        
        trainer.train()
        trainer.model.save_pretrained(output_dir)
        
        log("Done!")
