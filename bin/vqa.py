import json

from keras.preprocessing.text import Tokenizer

DICT_SIZE = 20

# --------------- CREATE DICTIONARY -----------------
tokenizer = Tokenizer(DICT_SIZE)

# Retrieve the questions from the dataset
q_data = json.load(open('../data/train/questions'))
questions = [q['question'].encode('utf8') for q in q_data['questions']]
# Retrieve the answers (a.k.a annotations) from the dataset
a_data = json.load(open('../data/train/annotations'))
answers = [ans['answer'].encode('utf8') for ann in a_data['annotations'] for ans in ann['answers']]

tokenizer.fit_on_texts(questions + answers)

