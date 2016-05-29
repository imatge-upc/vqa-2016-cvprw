import argparse
import cPickle as pickle
import sys

import h5py
import numpy as np
import os
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

sys.path.append('..')

from vqa.model.library import ModelLibrary
from vqa.dataset.dataset import VQADataset
from vqa.dataset.types import DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Constants
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
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(model_num):
    weights_path = WEIGHTS_DIR_PATH + 'model_weights_' + str(model_num) + '.{epoch:02d}.hdf5'
    losses_path = '../results/train_losses_{}.h5'.format(model_num)

    dataset = load_dataset()
    vqa_model = ModelLibrary.get_model(model_num, vocabulary_size=VOCABULARY_SIZE, embed_hidden_size=EMBED_HIDDEN_SIZE,
                                       question_max_len=dataset.question_max_len)
    train(vqa_model, dataset, model_num, weights_path, losses_path)


def load_dataset():
    if not os.path.isfile(DATASET_PREPROCESSED_PATH):
        print('Creating dataset...')
        dataset = VQADataset(DatasetType.TRAIN, QUESTIONS_PATH, ANNOTATIONS_PATH, RAW_DATASET_PREPROCESSED_PATH,
                             TOKENIZER_PATH, vocab_size=VOCABULARY_SIZE)
        print('Preparing dataset...')
        dataset.prepare()
        print('Dataset size: %d' % dataset.size())
        print('Dataset ready.')
        print('Saving dataset...')
        pickle.dump(dataset, open(DATASET_PREPROCESSED_PATH, 'w'))
        print('Dataset saved')
    else:
        print('Loading dataset...')
        dataset = pickle.load(open(DATASET_PREPROCESSED_PATH, 'r'))
        print('Dataset loaded')

    return dataset


def train(model, dataset, model_num, model_weights_path, losses_path):
    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, model_num, monitor='loss')
    stop_callback = EarlyStopping(monitor='loss', patience=5, mode='min')

    print('Start training...')
    model.fit_generator(dataset.batch_generator(BATCH_SIZE), samples_per_epoch=dataset.size(), nb_epoch=NUM_EPOCHS,
                        callbacks=[save_weights_callback, loss_callback, stop_callback])
    print('Trained')


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    def __init__(self, losses_path):
        super(LossHistoryCallback, self).__init__()
        self.losses = []
        self.losses_path = losses_path

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        try:
            with h5py.File(self.losses_path, 'a') as f:
                if 'train_losses' in f:
                    del f['train_losses']
                f.create_dataset('train_losses', data=np.array(self.losses))
        except (TypeError, RuntimeError):
            print('Couldnt save losses')


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, model_num, monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):
        super(CustomModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                    save_best_only=save_best_only, mode=mode)
        self.model_num = model_num

    def on_train_end(self, logs={}):
        os.symlink(WEIGHTS_DIR_PATH + 'model_weights_{}.{}.hdf5'.format(self.model_num, NUM_EPOCHS - 1),
                   WEIGHTS_DIR_PATH + 'model_weights_{}'.format(self.model_num))


# ------------------------------- ENTRY POINT -------------------------------

def check_model_num(model_num_str):
    model_num = int(model_num_str)
    if not ModelLibrary.is_valid_model_num(model_num):
        raise argparse.ArgumentTypeError('{} is not a defined model number'.format(model_num))

    return model_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument(
        'model',
        type=check_model_num,
        help='Specify the model number to be trained. Each model architecture has a model number associated'
    )
    # Start script
    main(parser.parse_args().model)
