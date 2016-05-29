import cPickle as pickle
import json
import sys
import timeit

import numpy as np
import os
from keras.layers import Input, Embedding, merge, LSTM, Dropout, Dense, RepeatVector
from keras.models import Model, model_from_json

sys.path.append('..')

from vqa.dataset.dataset import VQADataset, DatasetType

# ------------------------------ CONSTANTS ------------------------------
# Paths
DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH + 'preprocessed/'
DATASET_PREPROCESSED_PATH = PREPROC_DATA_PATH + 'test_dataset.p'
MODELS_DIR_PATH = '../models/'
MODEL_PATH = MODELS_DIR_PATH + 'model.json'
WEIGHTS_DIR_PATH = MODELS_DIR_PATH + 'weights/'
MODEL_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'model_weights.hdf5'
VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = WEIGHTS_DIR_PATH + 'truncated_vgg16_weights.h5'
RESULTS_PATH = '../results/test2015_results.json'
# Constants
VOCABULARY_SIZE = 20000
QUESTION_MAX_LEN = 22
EMBED_HIDDEN_SIZE = 100
NUM_EPOCHS = 40
BATCH_SIZE = 128

# --------------- CREATE DATASET -----------------
if not os.path.isfile(DATASET_PREPROCESSED_PATH):
    print('Creating dataset...')
    start_time = timeit.default_timer()
    dataset = VQADataset(DatasetType.TEST, '../data/test/questions', None, '../data/preprocessed/dataset/',
                         '../data/preprocessed/tokenizer.p', vocab_size=VOCABULARY_SIZE)
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
    image_repeat = RepeatVector(n=QUESTION_MAX_LEN)(image_input)

    # Question
    question_input = Input(shape=(QUESTION_MAX_LEN,), dtype='int32')
    question_embedded = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=QUESTION_MAX_LEN)(question_input)  # Can't use masking
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
