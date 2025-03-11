from dataclasses import dataclass

@dataclass
class DataConfig:
    """
    Represents the configuration for dataset processing and manipulation.

    This class is a data structure to hold configuration values required for
    preparing and transforming a dataset. It includes specifications for the
    dataset location, testing split parameters, randomization seed, and criteria
    for dimensionality reduction.

    :ivar dataset_path: Path to the dataset file.
    :type dataset_path: str
    :ivar test_size: Proportion of the dataset to include in the test split.
    :type test_size: float
    :ivar random_state: Seed value for reproducibility of random operations.
    :type random_state: int
    :ivar n_components: Percentage of variance to retain during dimensionality
        reduction.
    :type n_components: float
    """
    dataset_path: str
    test_size: float
    random_state: int
    n_components: float # percentage of variance to retain