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
        load_from_checkpoint: bool,
        start_num: int,
        end_num: int
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
    _, _, test_data = data_preparation(config, recbole_dataset)


    log(f"""
        ---------------------
        Config: 
            Model Name = {model_name}
            Dataset Name = {dataset_name}
            Start Index = {start_num}
            End Index = {end_num}
            Number of Recommendations = {config[KEYWORDS.NUMBER_OF_RECOMS]}
            Number of History Items = {config[KEYWORDS.NUMBER_OF_HISTORY_ITEMS]}
            Temperature = {config[KEYWORDS.TEMPERATURE]}
            Max Tokens = {config[KEYWORDS.MAX_TOKENS]}
            Top-K = {config[KEYWORDS.TOP_K]}
            Top-P = {config[KEYWORDS.TOP_P]}
        ---------------------
        """)
    
    model = model_class(
        config, 
        recbole_dataset, 
        load_from_checkpoint=load_from_checkpoint, 
        cls=DatasetNameEnum.get_dataset_type_cls(dataset_enum_name)
    ).to(config[KEYWORDS.DEVICE])

    trainer = LLMBasedTrainer(config, model, recbole_dataset)
    _ = trainer.evaluate(test_data, start_num=start_num, end_num=end_num, show_progress=config[KEYWORDS.SHOW_PROGRESS])

    log("Done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="GenRec", help='The model name to run')
    parser.add_argument('--dataset_name', type=str, default="AMAZON_TOY_GAMES", help='It should be chosen from DatasetNameEnum')
    parser.add_argument('--start_index', type=int, default=0, help='Start Index of Data')
    parser.add_argument('--end_index', type=int, default=10, help='End Index of Data')

    args = parser.parse_args()
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    start_index = args.start_index
    end_index = args.end_index
    
    execute(
        model_name=model_name, 
        dataset_enum_name= dataset_name, 
        load_from_checkpoint=True,
        start_num=start_index,
        end_num=end_index
    )
