import cPickle as pickle
import json
import sys
import timeit

import numpy as np
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
TRAIN_DIR_PATH = DATA_PATH + 'test/'
QUESTIONS_PATH = TRAIN_DIR_PATH + 'questions'
DATA_PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'test_dataset.p'
RAW_DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'dataset/'
TOKENIZER_PATH = DATA_PREPROCESSED_PATH + 'tokenizer.p'
WEIGHTS_DIR_PATH = '../models/weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(MODEL_NUM)
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
RESULTS_PATH = '../results/test2015_results_{}.json'.format(MODEL_NUM)

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.TEST, QUESTIONS_PATH, None, RAW_DATASET_PREPROCESSED_PATH, TOKENIZER_PATH,
                         vocab_size=VOCABULARY_SIZE)
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

# ------------------------------- TEST MODEL ----------------------------------
print('Loading weights...')
vqa_model.load_weights(MODEL_WEIGHTS_PATH)
print('Weights loaded')
print('Predicting...')
images, questions = dataset.get_dataset_input()
results = vqa_model.predict([images, questions], BATCH_SIZE)
print('Answers predicted')

print('Transforming results...')
results = np.argmax(results, axis=1)  # Max index evaluated on rows (1 row = 1 sample)
results = list(results)
print('Results transformed')

print('Building reverse word dictionary...')
word_dict = {idx: word for word, idx in dataset.tokenizer.word_index.iteritems()}
print('Reverse dictionary build')

print('Saving results...')
results_dict = [{'answer': word_dict[results[idx]], 'question_id': sample.question.id}
                for idx, sample in enumerate(dataset.samples)]
with open(RESULTS_PATH, 'w') as f:
    json.dump(results_dict, f)
print('Results saved')
