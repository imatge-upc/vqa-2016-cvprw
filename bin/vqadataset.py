import sys

sys.path.append('..')

from vqa.dataset.dataset import VQADataset, DatasetType

dataset = VQADataset(DatasetType.TRAIN, '../data/train/questions', '../data/train/annotations',
                     '../data/train/images/', '../data/preprocessed/preprocessed_dataset.h5',
                     '../data/preprocessed/tokenizer.p')
dataset.prepare()
