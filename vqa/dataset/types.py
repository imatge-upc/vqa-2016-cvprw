from enum import Enum


class DatasetType(Enum):
    """Enumeration with the possible sample types"""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2
