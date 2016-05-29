import cPickle as pickle
import sys
import timeit

import os

sys.path.append('..')

from vqa.model.library import ModelLibrary
from vqa.dataset.dataset import VQADataset, DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Constants
MODEL_NUM = ModelLibrary.MODEL_TWO
VOCABULARY_SIZE = 20000
EMBED_HIDDEN_SIZE = 100
NUM_EPOCHS = 40
BATCH_SIZE = 128

# Paths
DATA_PATH = '../data/'
TRAIN_DIR_PATH = DATA_PATH + 'val/'
QUESTIONS_PATH = TRAIN_DIR_PATH + 'questions'
ANNOTATIONS_PATH = TRAIN_DIR_PATH + 'annotations'
DATA_PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'validate_dataset.p'
RAW_DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'dataset/'
TOKENIZER_PATH = DATA_PREPROCESSED_PATH + 'tokenizer.p'
WEIGHTS_DIR_PATH = '../models/weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(MODEL_NUM)
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.VALIDATION, QUESTIONS_PATH, ANNOTATIONS_PATH, RAW_DATASET_PREPROCESSED_PATH,
                         TOKENIZER_PATH, vocab_size=VOCABULARY_SIZE)
    print('Preparing dataset...')
    dataset.prepare()
    elapsed_time = timeit.default_timer() - start_time
    print('Dataset size: %d' % dataset.size())
    print('Dataset ready. Execution time: %f' % elapsed_time)
    print('Saving dataset...')
    start_time = timeit.default_timer()
    pickle.dump(dataset, open(DATASET_PREPROCESSED_PATH, 'w'))
    elapsed_time = timeit.default_timer() - start_time
    print('Dataset saved. Execution time: %f' % elapsed_time)
else:
    print('Loading dataset...')
    start_time = timeit.default_timer()
    dataset = pickle.load(open(DATASET_PREPROCESSED_PATH, 'r'))
    elapsed_time = timeit.default_timer() - start_time
    print('Dataset loaded. Execution time: %f' % elapsed_time)

# ------------------------------- CREATE MODEL ----------------------------------

vqa_model = ModelLibrary.get_model(MODEL_NUM, vocabulary_size=VOCABULARY_SIZE, embed_hidden_size=EMBED_HIDDEN_SIZE,
                                   question_max_len=dataset.question_max_len)

# ------------------------------- VALIDATE MODEL ----------------------------------
print('Loading weights...')
vqa_model.load_weights(MODEL_WEIGHTS_PATH)
print('Weights loaded')
print('Start validation...')
result = vqa_model.evaluate_generator(dataset.batch_generator(BATCH_SIZE), val_samples=dataset.size())
print('Validated')
print(result)
