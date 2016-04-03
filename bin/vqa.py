import json
import os
import cPickle as pickle

from keras.preprocessing.text import Tokenizer

# ------------------------------ CONSTANTS ------------------------------

VOCABULARY_SIZE = 20000
PREPROC_DATA_PATH = '../data/preprocessed/'
TOKENIZER_PATH = PREPROC_DATA_PATH + 'tokenizer.p'
QUESTIONS_TOKENS_PATH = PREPROC_DATA_PATH + 'question_tokens.json'
ANSWERS_TOKENS_PATH = PREPROC_DATA_PATH + 'answer_tokens.json'

# --------------- CREATE DICTIONARY & TOKENIZER -----------------
# Retrieve the questions from the dataset
q_data = json.load(open('../data/train/questions'))
questions = [q['question'].encode('utf8') for q in q_data['questions']]
# Retrieve the answers (a.k.a annotations) from the dataset
a_data = json.load(open('../data/train/annotations'))
answers = [ans['answer'].encode('utf8') for ann in a_data['annotations'] for ans in ann['answers']]

# Retrieve tokenizer (create or load)
if not os.path.isfile(TOKENIZER_PATH):
    tokenizer = Tokenizer(VOCABULARY_SIZE)

    tokenizer.fit_on_texts(questions + answers)

    # print('\n'.join('{}: {}'.format(key, value) for key, value in tokenizer.word_index.iteritems()))

    if not os.path.isdir(PREPROC_DATA_PATH):
        os.mkdir(PREPROC_DATA_PATH)

    pickle.dump(tokenizer, open(TOKENIZER_PATH, 'w'))

else:
    tokenizer = pickle.load(open(TOKENIZER_PATH, 'r'))


# ------------------------------- TOKENIZE DATA ---------------------------------
# Get questions tokens
if not os.path.isfile(QUESTIONS_TOKENS_PATH):
    questions_tokens = tokenizer.texts_to_sequences(questions)
    json.dump(questions_tokens, open(QUESTIONS_TOKENS_PATH, 'w'))
else:
    questions_tokens = json.load(open(QUESTIONS_TOKENS_PATH, 'r'))

# Get answers tokens
if not os.path.isfile(ANSWERS_TOKENS_PATH):
    answers_tokens = tokenizer.texts_to_sequences(answers)
    json.dump(answers_tokens, open(ANSWERS_TOKENS_PATH, 'w'))
else:
    answers_tokens = json.load(open(ANSWERS_TOKENS_PATH, 'r'))

query_maxlen = max(map(len, (question for question in questions_tokens)))
