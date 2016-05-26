import cPickle as pickle
import sys
import timeit

import h5py
import numpy as np
import os
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, merge, LSTM, Dropout, Dense, RepeatVector
from keras.models import Model, model_from_json

sys.path.append('..')

from vqa.dataset.dataset import VQADataset
from vqa.dataset.types import DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Paths
DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = PREPROC_DATA_PATH + 'train_dataset.p'
MODELS_DIR_PATH = '../models/'
MODEL_PATH = MODELS_DIR_PATH + 'model.json'
WEIGHTS_DIR_PATH = MODELS_DIR_PATH + 'weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights.{epoch:02d}.hdf5'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
# Constants
VOCABULARY_SIZE = 20000
EMBED_HIDDEN_SIZE = 100
NUM_EPOCHS = 40
BATCH_SIZE = 128

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.TRAIN, '../data/train/questions', '../data/train/annotations',
                         '../data/preprocessed/dataset/', '../data/preprocessed/tokenizer.p',
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

# ------------------------------- CREATE MODEL -------------------------------

try:
    with open(MODEL_PATH, 'r') as f:
        print('Loading model...')
        vqa_model = model_from_json(f.read())
        print('Model loaded')
        print('Compiling model...')
        vqa_model.compile(optimizer='adam', loss='categorical_crossentropy')
        print('Model compiled')
except IOError:
    print('Creating model...')
    # Image
    image_input = Input(shape=(1024,))
    image_repeat = RepeatVector(n=dataset.question_max_len)(image_input)

    # Question
    question_input = Input(shape=(dataset.question_max_len,), dtype='int32')
    question_embedded = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=dataset.question_max_len)(question_input)  # Can't use masking
    question_embedded = Dropout(0.5)(question_embedded)

    # Merge
    merged = merge([image_repeat, question_embedded], mode='concat')  # Merge for layers merge for tensors
    x = LSTM(EMBED_HIDDEN_SIZE, return_sequences=False)(merged)
    x = Dropout(0.5)(x)
    output = Dense(output_dim=VOCABULARY_SIZE, activation='softmax')(x)

    vqa_model = Model(input=[image_input, question_input], output=output)
    print('Model created')

    print('Compiling model...')
    vqa_model.compile(optimizer='adam', loss='categorical_crossentropy')
    print('Model compiled')

    print('Saving model...')
    model_json = vqa_model.to_json()
    with open(MODEL_PATH, 'w') as f:
        f.write(model_json)
    print('Model saved')


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    def __init__(self):
        super(LossHistoryCallback, self).__init__()
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        try:
            with h5py.File('../results/train_losses.h5', 'a') as f:
                if 'train_losses' in f:
                    del f['train_losses']
                f.create_dataset('train_losses', data=np.array(self.losses))
        except (TypeError, RuntimeError):
            print('Couldnt save losses')


loss_callback = LossHistoryCallback()
save_weights_callback = ModelCheckpoint(MODEL_WEIGHTS_PATH, monitor='loss')
stop_callback = EarlyStopping(monitor='loss', patience=5, mode='min')

# ------------------------------- TRAIN MODEL -------------------------------
print('Start training...')
vqa_model.fit_generator(dataset.batch_generator(BATCH_SIZE), samples_per_epoch=dataset.size(), nb_epoch=NUM_EPOCHS,
                        callbacks=[save_weights_callback, loss_callback, stop_callback])
print('Trained')

