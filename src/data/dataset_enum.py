from enum import Enum

# Choose the dataset name from here: 
# recbole/properties/dataset/url.yaml
class DatasetNameEnum(Enum):
    MOVIE_LENS = 'ml-1m'
    AMAZON_TOY_GAMES = 'amazon-toys-games'

    @staticmethod
    def get_dataset_name(name: str):
        try:
            return DatasetNameEnum[name].value
        except KeyError:
            raise ValueError(f"This name is not supported as a dataset: {name}")

