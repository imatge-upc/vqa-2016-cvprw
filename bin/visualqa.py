import argparse
import cPickle as pickle
import json
import sys

import h5py
import numpy as np
import os
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

sys.path.append('..')

from vqa.dataset.types import DatasetType
from vqa.dataset.dataset import VQADataset

from vqa.model.library import ModelLibrary

# ------------------------------ GLOBALS ------------------------------
# Constants
ACTIONS = ['train', 'val', 'test', 'eval']
VOCABULARY_SIZE = 20000
NUM_EPOCHS = 40
BATCH_SIZE = 128

# Paths
DATA_PATH = '../data/'
PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
TOKENIZER_PATH = PREPROCESSED_PATH + 'tokenizer.p'
FEATURES_DIR_PATH = PREPROCESSED_PATH + 'features/'
WEIGHTS_DIR_PATH = '../models/weights/'
RESULTS_DIR_PATH = '../results/'

# Config
CONFIG_TRAIN = {
    'dataset_type': DatasetType.TRAIN,
    'dataset_path': PREPROCESSED_PATH + 'train_dataset.p',
    'questions_path': DATA_PATH + 'train/questions',
    'annotations_path': DATA_PATH + 'train/annotations'
}
CONFIG_VAL = {
    'dataset_type': DatasetType.VALIDATION,
    'dataset_path': PREPROCESSED_PATH + 'validate_dataset.p',
    'questions_path': DATA_PATH + 'val/questions',
    'annotations_path': DATA_PATH + 'val/annotations'
}
CONFIG_TEST = {
    'dataset_type': DatasetType.TEST,
    'dataset_path': PREPROCESSED_PATH + 'test_dataset.p',
    'questions_path': DATA_PATH + 'test/questions',
    'annotations_path': None
}
CONFIG_EVAL = {
    'dataset_type': DatasetType.EVAL,
    'dataset_path': PREPROCESSED_PATH + 'eval_dataset.p',
    'questions_path': DATA_PATH + 'val/questions',
    'annotations_path': None
}

# Defaults
DEFAULT_MODEL = max(ModelLibrary.get_valid_model_nums())
DEFAULT_ACTION = 'train'


# ------------------------------- SCRIPT FUNCTIONALITY -------------------------------

def main(action, model_num):
    print('Action: ' + action)
    print('Model number: {}'.format(model_num))

    # Always load train dataset to obtain the question_max_len from it
    train_dataset = load_dataset(CONFIG_TRAIN['dataset_type'], CONFIG_TRAIN['dataset_path'],
                                 CONFIG_TRAIN['questions_path'], CONFIG_TRAIN['annotations_path'], FEATURES_DIR_PATH,
                                 TOKENIZER_PATH)
    question_max_len = train_dataset.question_max_len

    # Load model
    vqa_model = ModelLibrary.get_model(model_num, vocabulary_size=VOCABULARY_SIZE, question_max_len=question_max_len)

    # Load dataset depending on the action to perform
    if action == 'train':
        dataset = train_dataset
        val_dataset = load_dataset(CONFIG_VAL['dataset_type'], CONFIG_VAL['dataset_path'],
                                   CONFIG_VAL['questions_path'], CONFIG_VAL['annotations_path'], FEATURES_DIR_PATH,
                                   TOKENIZER_PATH)
        weights_path = WEIGHTS_DIR_PATH + 'model_weights_' + str(model_num) + '.{epoch:02d}.hdf5'
        losses_path = RESULTS_DIR_PATH + 'losses_{}.h5'.format(model_num)
        train(vqa_model, dataset, model_num, weights_path, losses_path, val_dataset)
    elif action == 'val':
        dataset = load_dataset(CONFIG_VAL['dataset_type'], CONFIG_VAL['dataset_path'],
                               CONFIG_VAL['questions_path'], CONFIG_VAL['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH)
        weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
        validate(vqa_model, dataset, weights_path)
    elif action == 'test':
        dataset = load_dataset(CONFIG_TEST['dataset_type'], CONFIG_TEST['dataset_path'],
                               CONFIG_TEST['questions_path'], CONFIG_TEST['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH)
        weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
        results_path = RESULTS_DIR_PATH + 'test2015_results_{}.json'.format(model_num)
        test(vqa_model, dataset, weights_path, results_path)
    elif action == 'eval':
        dataset = load_dataset(CONFIG_EVAL['dataset_type'], CONFIG_EVAL['dataset_path'],
                               CONFIG_EVAL['questions_path'], CONFIG_EVAL['annotations_path'], FEATURES_DIR_PATH,
                               TOKENIZER_PATH)
        weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
        results_path = RESULTS_DIR_PATH + 'val2014_results_{}.json'.format(model_num)
        test(vqa_model, dataset, weights_path, results_path)
    else:
        raise ValueError('The action you provided do not exist')


def load_dataset(dataset_type, dataset_path, questions_path, annotations_path, features_dir_path, tokenizer_path):
    try:
        with open(dataset_path, 'r') as f:
            print('Loading dataset...')
            dataset = pickle.load(f)
            print('Dataset loaded')
    except IOError:
        print('Creating dataset...')
        dataset = VQADataset(dataset_type, questions_path, annotations_path, features_dir_path,
                             tokenizer_path, vocab_size=VOCABULARY_SIZE)
        print('Preparing dataset...')
        dataset.prepare()
        print('Dataset size: %d' % dataset.size())
        print('Dataset ready.')

        print('Saving dataset...')
        with open(dataset_path, 'w') as f:
            pickle.dump(dataset, f)
        print('Dataset saved')

    return dataset


def train(model, dataset, model_num, model_weights_path, losses_path, val_dataset):
    loss_callback = LossHistoryCallback(losses_path)
    save_weights_callback = CustomModelCheckpoint(model_weights_path, WEIGHTS_DIR_PATH, model_num)
    # TODO: add the early stopping again
    # stop_callback = EarlyStopping(patience=5)

    print('Start training...')
    model.fit_generator(dataset.batch_generator(BATCH_SIZE), samples_per_epoch=dataset.size(), nb_epoch=NUM_EPOCHS,
                        callbacks=[save_weights_callback, loss_callback],
                        validation_data=val_dataset.batch_generator(BATCH_SIZE), nb_val_samples=val_dataset.size())
    print('Trained')


def validate(model, dataset, weights_path):
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Start validation...')
    result = model.evaluate_generator(dataset.batch_generator(BATCH_SIZE), val_samples=dataset.size())
    print('Validated')

    return result


def test(model, dataset, weights_path, results_path):
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Predicting...')
    images, questions = dataset.get_dataset_input()
    results = model.predict([images, questions], BATCH_SIZE)
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
    with open(results_path, 'w') as f:
        json.dump(results_dict, f)
    print('Results saved')


# ------------------------------- CALLBACKS -------------------------------

class LossHistoryCallback(Callback):
    def __init__(self, results_path):
        super(LossHistoryCallback, self).__init__()
        self.train_losses = []
        self.val_losses = []
        self.results_path = results_path

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        try:
            with h5py.File(self.results_path, 'a') as f:
                if 'train_losses' in f:
                    del f['train_losses']
                if 'val_losses' in f:
                    del f['val_losses']
                f.create_dataset('train_losses', data=np.array(self.train_losses))
                f.create_dataset('val_losses', data=np.array(self.val_losses))
        except (TypeError, RuntimeError):
            print('Couldn\'t save losses')


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, weights_dir_path, model_num, monitor='val_loss', verbose=0, save_best_only=False,
                 mode='auto'):
        super(CustomModelCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                    save_best_only=save_best_only, mode=mode)
        self.model_num = model_num
        self.weights_dir_path = weights_dir_path
        self.last_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        super(CustomModelCheckpoint, self).on_epoch_end(epoch, logs)
        self.last_epoch = epoch

    def on_train_end(self, logs={}):
        try:
            os.symlink(self.weights_dir_path + 'model_weights_{}.{}.hdf5'.format(self.model_num, self.last_epoch),
                       self.weights_dir_path + 'model_weights_{}'.format(self.model_num))
        except OSError:
            # If the symlink already exist, delete and create again
            os.remove(self.weights_dir_path + 'model_weights_{}'.format(self.model_num))
            # Recreate
            os.symlink(self.weights_dir_path + 'model_weights_{}.{}.hdf5'.format(self.model_num, self.last_epoch),
                       self.weights_dir_path + 'model_weights_{}'.format(self.model_num))
            pass


# ------------------------------- ENTRY POINT -------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main entry point to interact with the VQA module')
    parser.add_argument(
        '-m',
        '--model',
        type=int,
        choices=ModelLibrary.get_valid_model_nums(),
        default=DEFAULT_MODEL,
        help='Specify the model architecture to interact with. Each model architecture has a model number associated.'
             'By default, the model will be the last architecture created, i.e., the model with the biggest number'
    )
    parser.add_argument(
        '-a',
        '--action',
        choices=ACTIONS,
        default=DEFAULT_ACTION,
        help='Which action should be perform on the model. By default, training will be done'
    )
    # Start script
    args = parser.parse_args()
    main(args.action, args.model)
