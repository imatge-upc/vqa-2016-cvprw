import sys
import os
import h5py
import shutil
import timeit
import cPickle as pickle
import json

from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, RepeatVector, Merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM

sys.path.append('..')

from vqa.dataset.dataset import VQADataset, DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Paths
DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH + 'preprocessed/'
TRAIN_IMAGES_PATH = DATA_PATH + 'train/images/'
LIST_IMAGES_FILE_PATH = TRAIN_IMAGES_PATH + 'list_images.json'
TOKENIZER_PATH = PREPROC_DATA_PATH + 'tokenizer.p'
DATASET_PREPROCESSED_PATH = PREPROC_DATA_PATH + 'train_dataset.p'
MODELS_DIR_PATH = '../models/'
MODEL_PATH = MODELS_DIR_PATH + 'model.json'
WEIGHTS_DIR_PATH = MODELS_DIR_PATH + 'weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
# Constants
VOCABULARY_SIZE = 20000
EMBED_HIDDEN_SIZE = 100
NUM_EPOCHS = 40
BATCH_SIZE = 32

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.TRAIN, '../data/train/questions', '../data/train/annotations',
                         '../data/train/images/', '../data/preprocessed/tokenizer.p', vocab_size=VOCABULARY_SIZE)
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
if not os.path.isfile(MODEL_PATH):
    print('Creating model...')
    start_time = timeit.default_timer()
    # Question
    query_model = Sequential()
    query_model.add(
        Embedding(VOCABULARY_SIZE, EMBED_HIDDEN_SIZE, input_length=dataset.question_max_len, mask_zero=True))
    query_model.add(Dropout(0.5))

    # Image
    # Set to all layers trainable=False to freeze VGG-16 weights'
    im_model = Sequential()
    im_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224), trainable=False))
    im_model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    im_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    im_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    im_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(ZeroPadding2D((1, 1), trainable=False))
    im_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    im_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    im_model.add(Flatten(trainable=False))

    # Load VGG-16 weights
    if not os.path.isfile(VGG_WEIGHTS_PATH):
        print('You need to download the VGG-16 weights: '
              'https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?pref=2&pli=1')
        quit()

    if not os.path.isfile(TRUNCATED_VGG_WEIGHTS_PATH):
        print('Preparing VGG-16 weights...')
        shutil.copy(VGG_WEIGHTS_PATH, TRUNCATED_VGG_WEIGHTS_PATH)
        trunc_vgg_weights = h5py.File(TRUNCATED_VGG_WEIGHTS_PATH)
        # Remove last 5 layers' weights
        for i in range(32, 36):
            del trunc_vgg_weights['layer_{}'.format(i)]
        trunc_vgg_weights.attrs.modify('nb_layers', 32)
        trunc_vgg_weights.flush()
        trunc_vgg_weights.close()
        print('VGG weights ready to use')

    im_model.load_weights(TRUNCATED_VGG_WEIGHTS_PATH)

    im_model.add(RepeatVector(dataset.question_max_len))

    # Merging
    model = Sequential()
    model.add(Merge([query_model, im_model], mode='concat'))
    model.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(VOCABULARY_SIZE, activation='softmax'))

    elapsed_time = timeit.default_timer() - start_time
    print('Model created. Execution time: %f' % elapsed_time)

    print('Compiling model...')
    start_time = timeit.default_timer()
    model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
    elapsed_time = timeit.default_timer() - start_time
    print('Model compiled. Execution time: %f' % elapsed_time)

    if not os.path.isdir(MODELS_DIR_PATH):
        os.mkdir(MODELS_DIR_PATH)

    print('Saving model...')
    start_time = timeit.default_timer()
    model_json = model.to_json()
    open(MODEL_PATH, 'w').write(model_json)
    elapsed_time = timeit.default_timer() - start_time
    print('Model saved. Execution time: %f' % elapsed_time)
else:
    print('Loading model...')
    start_time = timeit.default_timer()
    model = model_from_json(open(MODEL_PATH, 'r').read())
    elapsed_time = timeit.default_timer() - start_time
    print('Model loaded. Execution time: %f' % elapsed_time)


# ------------------------------- CALLBACKS ----------------------------------
class LossHistoryCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

loss_callback = LossHistoryCallback()
save_weights_callback = ModelCheckpoint(MODEL_WEIGHTS_PATH, monitor='loss', save_best_only=True, mode='min')
stop_callback = EarlyStopping(monitor='loss', patience=2, mode='min')

# ------------------------------- TRAIN MODEL ----------------------------------
print('Start training model...')
start_time = timeit.default_timer()
model.fit_generator(dataset.batch_generator(BATCH_SIZE), samples_per_epoch=dataset.size(), nb_epoch=NUM_EPOCHS,
                    callbacks=[loss_callback, save_weights_callback, stop_callback])
elapsed_time = timeit.default_timer() - start_time
print('Model trained. Execution time: %f' % elapsed_time)

print('Saving loss history...')
with open(MODELS_DIR_PATH + 'losses.json') as f:
    json.dump(loss_callback.losses, f)
print('Losses saved')


