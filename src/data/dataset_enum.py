from enum import Enum
from data.dataset import DatasetMovieLens, DatasetAmazon

# Choose the dataset name from here: 
# recbole/properties/dataset/url.yaml
class DatasetNameEnum(Enum):
    MOVIE_LENS = ('ml-1m', DatasetMovieLens)
    AMAZON_TOY_GAMES = ('amazon-toys-games', DatasetAmazon)

    @staticmethod
    def get_dataset_name(name: str):
        try:
            dataset_name, _ = DatasetNameEnum[name].value
            return dataset_name
        except KeyError:
            raise ValueError(f"This name is not supported as a dataset: {name}")

    @staticmethod
    def get_dataset_type_cls(name: str):
        try:
            _, dataset_type_cls = DatasetNameEnum[name].value
            return dataset_type_cls
        except KeyError:
            raise ValueError(f"This name is not supported as a dataset: {name}")

