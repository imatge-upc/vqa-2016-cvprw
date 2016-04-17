import json
import os
import cPickle as pickle
import numpy as np
import h5py
import shutil

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout, RepeatVector, Merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from scipy.misc import imread, imresize

# ------------------------------ CONSTANTS ------------------------------
# Paths
DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH + 'preprocessed/'
TRAIN_IMAGES_PATH = DATA_PATH + 'train/images/'
LIST_IMAGES_FILE_PATH = TRAIN_IMAGES_PATH + 'list_images.json'
TOKENIZER_PATH = PREPROC_DATA_PATH + 'tokenizer.p'
QUESTIONS_TOKENS_PATH = PREPROC_DATA_PATH + 'question_tokens.json'
ANSWERS_TOKENS_PATH = PREPROC_DATA_PATH + 'answer_tokens.json'
MODELS_DIR_PATH = '../models/'
MODEL_PATH = MODELS_DIR_PATH + 'model.json'
VGG_WEIGHTS_PATH = MODELS_DIR_PATH + 'vgg16_weights.h5'
TRUNCATED_VGG_WEIGHTS_PATH = MODELS_DIR_PATH + 'truncated_vgg16_weights.h5'
# Constants
VOCABULARY_SIZE = 20000
EMBED_HIDDEN_SIZE = 50
NUM_EPOCHS = 40

# --------------- CREATE DICTIONARY & TOKENIZER -----------------
# Retrieve the questions from the dataset
print('Loading questions...')
q_data = json.load(open('../data/train/questions'))
questions = [q['question'].encode('utf8') for q in q_data['questions']]
print('Questions loaded')
# Retrieve the answers (a.k.a annotations) from the dataset
print('Loading answers...')
a_data = json.load(open('../data/train/annotations'))
answers = [ans['answer'].encode('utf8') for ann in a_data['annotations'] for ans in ann['answers']]
print('Answers loaded')

# Retrieve tokenizer (create or load)
if not os.path.isfile(TOKENIZER_PATH):
    print('Creating tokenizer...')
    tokenizer = Tokenizer(VOCABULARY_SIZE)
    print('Feeding tokenizer with dataset...')
    tokenizer.fit_on_texts(questions + answers)
    print('Tokenizer created')

    # print('\n'.join('{}: {}'.format(key, value) for key, value in tokenizer.word_index.iteritems()))

    if not os.path.isdir(PREPROC_DATA_PATH):
        os.mkdir(PREPROC_DATA_PATH)

    print('Saving tokenizer...')
    pickle.dump(tokenizer, open(TOKENIZER_PATH, 'w'))
    print('Tokenizer saved')

else:
    print('Loading tokenizer...')
    tokenizer = pickle.load(open(TOKENIZER_PATH, 'r'))
    print('Tokenizer loaded')

# ------------------------------- TOKENIZE DATA ---------------------------------
# Get questions tokens
if not os.path.isfile(QUESTIONS_TOKENS_PATH):
    print('Tokenizing questions...')
    questions_tokens = tokenizer.texts_to_sequences(questions)
    print('Saving tokenized questions...')
    json.dump(questions_tokens, open(QUESTIONS_TOKENS_PATH, 'w'))
    print('Question tokens created and saved')
else:
    print('Loading tokenized questions...')
    questions_tokens = json.load(open(QUESTIONS_TOKENS_PATH, 'r'))
    print('Question tokens loaded')

# Get answers tokens
if not os.path.isfile(ANSWERS_TOKENS_PATH):
    print('Tokenizing answers...')
    answers_tokens = tokenizer.texts_to_sequences(answers)
    print('Saving tokenized answers...')
    json.dump(answers_tokens, open(ANSWERS_TOKENS_PATH, 'w'))
    print('Answer tokens created and saved')
else:
    print('Loading tokenized answers...')
    answers_tokens = json.load(open(ANSWERS_TOKENS_PATH, 'r'))
    print('Answer tokens loaded')

query_maxlen = max(map(len, (question for question in questions_tokens)))

# ------------------------------- CREATE MODEL ----------------------------------
if not os.path.isfile(MODEL_PATH):
    print('Creating model...')
    # Question
    query_model = Sequential()
    query_model.add(Embedding(VOCABULARY_SIZE, EMBED_HIDDEN_SIZE, input_length=query_maxlen, mask_zero=True))
    query_model.add(Dropout(0.3))

    # Image
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

    im_model.load_weights(TRUNCATED_VGG_WEIGHTS_PATH)

    im_model.add(RepeatVector(query_maxlen))

    # Merging
    model = Sequential()
    model.add(Merge([query_model, im_model], mode='concat'))
    model.add(LSTM(EMBED_HIDDEN_SIZE, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(VOCABULARY_SIZE, activation='softmax'))

    print('Model created')

    print('Compiling model...')
    model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
    print('Model compiled')

    if not os.path.isdir(MODELS_DIR_PATH):
        os.mkdir(MODELS_DIR_PATH)

    print('Saving model...')
    model_json = model.to_json()
    open(MODEL_PATH, 'w').write(model_json)
    print('Model saved')
else:
    print('Loading model...')
    model = model_from_json(open(MODEL_PATH, 'r').read())
    print('Model loaded')
