import cPickle as pickle
import json
import sys
import timeit

import os
from keras.callbacks import Callback

sys.path.append('..')

from vqa.dataset.dataset import VQADataset, DatasetType
from vqa.model.model import VQAModel

# ------------------------------ CONSTANTS ------------------------------
# Paths
DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = PREPROC_DATA_PATH + 'validate_dataset.p'
MODELS_DIR_PATH = '../models/'
MODEL_PATH = MODELS_DIR_PATH + 'model.json'
WEIGHTS_DIR_PATH = MODELS_DIR_PATH + 'weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
# Constants
VOCABULARY_SIZE = 20000
NUM_EPOCHS = 40
BATCH_SIZE = 32

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.VALIDATION, '../data/val/questions', '../data/val/annotations',
                         '../data/val/images/', '../data/preprocessed/tokenizer.p', vocab_size=VOCABULARY_SIZE)
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
model = VQAModel(MODEL_PATH, dataset.question_max_len, VOCABULARY_SIZE, VGG_WEIGHTS_PATH, TRUNCATED_VGG_WEIGHTS_PATH,
                 MODEL_WEIGHTS_PATH)
model.prepare()


# ------------------------------- CALLBACKS ----------------------------------
class LossHistoryCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


loss_callback = LossHistoryCallback()

# ------------------------------- VALIDATE MODEL ----------------------------------
history = model.train(dataset, NUM_EPOCHS, BATCH_SIZE, [loss_callback])

print('Saving history...')
with open(MODELS_DIR_PATH + 'validate_history.json') as f:
    json.dump(history, f)
print('History saved')
