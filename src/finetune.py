from recbole_trainer import LLMBasedTrainer
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from helpers.utils_general import get_absolute_path
from helpers.utils_global import *
from helpers.utils_recbole import get_model, create_config
from data.dataset_enum import DatasetNameEnum


def execute(
        model_name: str, 
        dataset_enum_name: str, 
        load_from_checkpoint: bool
    ):
    model_class = get_model(model_name)
    dataset_name = DatasetNameEnum.get_dataset_name(dataset_enum_name)

    props_dir = get_absolute_path("props")
    props = [
        f'{props_dir}/{model_name}.yaml', 
        f'{props_dir}/{dataset_name}.yaml', 
        f'{props_dir}/openai_api.yaml', 
        f'{props_dir}/overall.yaml'
    ]

    
    config = create_config(model_class, dataset_name, props)
    recbole_dataset = SequentialDataset(config)
    train_data, valid_data, _ = data_preparation(config, recbole_dataset)
    
    model = model_class(
        config, 
        recbole_dataset, 
        load_from_checkpoint=load_from_checkpoint, 
        cls=DatasetNameEnum.get_dataset_type_cls(dataset_enum_name)
    ).to(config[KEYWORDS.DEVICE])

    trainer = LLMBasedTrainer(config, model, recbole_dataset)
    _ = trainer.train(train_data, valid_data)
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="LLMRec", help='The model name to run')
    parser.add_argument('--dataset_name', type=str, default="MOVIE_LENS", help='It should be chosen from DatasetNameEnum')
    args = parser.parse_args()
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    
    execute(
        model_name=model_name, 
        dataset_enum_name=dataset_name, 
        load_from_checkpoint=False
    )