import sys

sys.path.append('..')

from vqa.dataset.dataset import VQADataset, DatasetType

dataset = VQADataset(DatasetType.TRAIN, '../data/train/questions', '../data/train/annotations',
                     '../data/train/images/', '../data/preprocessed/tokenizer.p', vocab_size=10)
dataset.prepare()
