from trainer import LLMBasedTrainer
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from helpers.utils_general import get_absolute_path
from helpers.utils_global import *
from helpers.utils_recbole import get_model, create_config
from data.dataset_enum import DatasetNameEnum


def execute(model_name: str, dataset_enum_name: str):
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

    model = model_class(
        config, 
        recbole_dataset, 
        load_from_checkpoint=True, 
        cls=DatasetNameEnum.get_dataset_type_cls(dataset_enum_name)
    ).to(config[KEYWORDS.DEVICE])

    trainer = LLMBasedTrainer(config, model, recbole_dataset)
    _ = trainer.evaluate(test_data, start_num=0, end_num=10, show_progress=config[KEYWORDS.SHOW_PROGRESS])

    log("Done!")


if __name__ == '__main__':
    model_name = "GenRec"
    dataset_enum_name = DatasetNameEnum.MOVIE_LENS.name
    execute(model_name, dataset_enum_name)
