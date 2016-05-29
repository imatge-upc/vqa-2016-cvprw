from enum import Enum


class DatasetType(Enum):
    """Enumeration with the possible sample types"""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    EVAL = 3            # Use validation set as if it was test set and predict andswers to check accuracy
