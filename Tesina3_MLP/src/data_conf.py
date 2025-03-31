from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_path: str
    test_size: float
    random_state: int
    n_components: float # percentage of variance to retain
