import argparse
import cPickle as pickle
import json
import sys

import numpy as np
import os

sys.path.append('..')

from vqa.model.library import ModelLibrary
from vqa.dataset.dataset import VQADataset, DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Constants
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
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'


def main(model_num):
    weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
    results_path = '../results/test2015_results_{}.json'.format(model_num)
    dataset = load_dataset()
    vqa_model = ModelLibrary.get_model(model_num, vocabulary_size=VOCABULARY_SIZE, embed_hidden_size=EMBED_HIDDEN_SIZE,
                                       question_max_len=dataset.question_max_len)
    test(vqa_model, dataset, weights_path, results_path)


def load_dataset():
    if not os.path.isfile(DATASET_PREPROCESSED_PATH):
        print('Creating dataset...')
        dataset = VQADataset(DatasetType.TEST, QUESTIONS_PATH, None, RAW_DATASET_PREPROCESSED_PATH, TOKENIZER_PATH,
                             vocab_size=VOCABULARY_SIZE)
        print('Preparing dataset...')
        dataset.prepare()
        print('Dataset size: %d' % dataset.size())
        print('Dataset ready')
        print('Saving dataset...')
        pickle.dump(dataset, open(DATASET_PREPROCESSED_PATH, 'w'))
        print('Dataset saved')
    else:
        print('Loading dataset...')
        dataset = pickle.load(open(DATASET_PREPROCESSED_PATH, 'r'))
        print('Dataset loaded')

    return dataset


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
