from trainer import LLMBasedTrainer
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from helpers.utils_general import get_absolute_path, ModelConfig, log
from helpers.utils_recbole import get_model
from prompts.prompts_general import LLAMA_PROMPT_FORMAT
from model.user_interaction import DatasetNameEnum, RecBoleItemTypeEnum

def create_config(model_class, dataset_name, props):
    from recbole.config import Config
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=None)
    return config


############################
# Load environment variables
from dotenv import dotenv_values
CONFIG = dotenv_values(get_absolute_path('.env.development'))

LLAMA2_CONFIG = ModelConfig(
    id="meta-llama/Llama-2-7b-chat-hf",
    model_short_name="llama2_7b",
    prompt_format=LLAMA_PROMPT_FORMAT,
    api_key=CONFIG["HF_API_KEY"],
    temperature=0.01, # The range is [0, 1]
    top_p=0.5,
    top_k=20,
    max_tokens=500,
)


model_name = "GenRec"

# Choose the dataset name from here: 
# recbole/properties/dataset/url.yaml

dataset_type = 'MOVIE_LENS'

dataset_name = DatasetNameEnum.get_dataset_name(dataset_type)
# dataset_interaction_type = RecBoleItemTypeEnum.get_class_by_name(dataset_type)

props_dir = get_absolute_path("props")
props = [
    f'{props_dir}/{model_name}.yaml', 
    f'{props_dir}/{dataset_name}.yaml', 
    f'{props_dir}/openai_api.yaml', 
    f'{props_dir}/overall.yaml'
]
print(props)

model_class = get_model(model_name)
config = create_config(model_class, dataset_name, props)

dataset = SequentialDataset(config)

train_data, valid_data, test_data = data_preparation(config, dataset)
model = model_class(config, dataset, LLAMA2_CONFIG, load_from_checkpoint=True).to(config['device'])
trainer = LLMBasedTrainer(config, model, dataset)
test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

log("Done!")