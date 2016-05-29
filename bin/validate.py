import argparse
import cPickle as pickle
import sys

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
TRAIN_DIR_PATH = DATA_PATH + 'val/'
QUESTIONS_PATH = TRAIN_DIR_PATH + 'questions'
ANNOTATIONS_PATH = TRAIN_DIR_PATH + 'annotations'
DATA_PREPROCESSED_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'validate_dataset.p'
RAW_DATASET_PREPROCESSED_PATH = DATA_PREPROCESSED_PATH + 'dataset/'
TOKENIZER_PATH = DATA_PREPROCESSED_PATH + 'tokenizer.p'
WEIGHTS_DIR_PATH = '../models/weights/'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'


def main(model_num):
    weights_path = WEIGHTS_DIR_PATH + 'model_weights_{}'.format(model_num)
    dataset = load_dataset()
    vqa_model = ModelLibrary.get_model(model_num, vocabulary_size=VOCABULARY_SIZE, embed_hidden_size=EMBED_HIDDEN_SIZE,
                                       question_max_len=dataset.question_max_len)
    result = validate(vqa_model, dataset, weights_path)
    print('Validation loss: {}'.format(result))


def load_dataset():
    if not os.path.isfile(DATASET_PREPROCESSED_PATH):
        print('Creating dataset...')
        dataset = VQADataset(DatasetType.VALIDATION, QUESTIONS_PATH, ANNOTATIONS_PATH, RAW_DATASET_PREPROCESSED_PATH,
                             TOKENIZER_PATH, vocab_size=VOCABULARY_SIZE)
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


def validate(model, dataset, weights_path):
    print('Loading weights...')
    model.load_weights(weights_path)
    print('Weights loaded')
    print('Start validation...')
    result = model.evaluate_generator(dataset.batch_generator(BATCH_SIZE), val_samples=dataset.size())
    print('Validated')

    return result


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
