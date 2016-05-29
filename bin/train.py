import cPickle as pickle
import sys
import timeit

import h5py
import numpy as np
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

sys.path.append('..')

from vqa.model.library import ModelLibrary
from vqa.dataset.dataset import VQADataset
from vqa.dataset.types import DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Constants
MODEL_NUM = ModelLibrary.MODEL_TWO
VOCABULARY_SIZE = 20000
EMBED_HIDDEN_SIZE = 100
NUM_EPOCHS = 40
BATCH_SIZE = 128

# Paths
DATA_PATH = '../data/'
TRAIN_DIR_PATH = DATA_PATH + 'train/'
QUESTIONS_PATH = TRAIN_DIR_PATH + 'questions'
ANNOTATIONS_PATH = TRAIN_DIR_PATH + 'annotations'
DATA_PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'train_dataset.p'
RAW_DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'dataset/'
TOKENIZER_PATH = DATA_PREPROCESSED_PATH + 'tokenizer.p'
WEIGHTS_DIR_PATH = '../models/weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights_' + str(MODEL_NUM) + '.{epoch:02d}.hdf5'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
LOSSES_PATH = '../results/train_losses_{}.h5'.format(MODEL_NUM)

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.TRAIN, QUESTIONS_PATH, ANNOTATIONS_PATH, RAW_DATASET_PREPROCESSED_PATH,
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

# ------------------------------- CREATE MODEL -------------------------------
vqa_model = ModelLibrary.get_model(MODEL_NUM, vocabulary_size=VOCABULARY_SIZE, embed_hidden_size=EMBED_HIDDEN_SIZE,
                                   question_max_len=dataset.question_max_len)


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    def __init__(self):
        super(LossHistoryCallback, self).__init__()
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        try:
            with h5py.File(LOSSES_PATH, 'a') as f:
                if 'train_losses' in f:
                    del f['train_losses']
                f.create_dataset('train_losses', data=np.array(self.losses))
        except (TypeError, RuntimeError):
            print('Couldnt save losses')


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):
        super(CustomModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                    save_best_only=save_best_only, mode=mode)

    def on_train_end(self, logs={}):
        os.symlink(WEIGHTS_DIR_PATH + 'model_weights_{}.{}.hdf5'.format(MODEL_NUM, NUM_EPOCHS - 1),
                   WEIGHTS_DIR_PATH + 'model_weights_{}'.format(MODEL_NUM))


loss_callback = LossHistoryCallback()
save_weights_callback = CustomModelCheckpoint(MODEL_WEIGHTS_PATH, monitor='loss')
stop_callback = EarlyStopping(monitor='loss', patience=5, mode='min')

# ------------------------------- TRAIN MODEL -------------------------------
print('Start training...')
vqa_model.fit_generator(dataset.batch_generator(BATCH_SIZE), samples_per_epoch=dataset.size(), nb_epoch=NUM_EPOCHS,
                        callbacks=[save_weights_callback, loss_callback, stop_callback])
print('Trained')
