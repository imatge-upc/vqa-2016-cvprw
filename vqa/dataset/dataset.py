import os
import json

import cPickle as pickle
from keras.preprocessing.text import Tokenizer

from sample import Question, Answer, Image, VQASample
from types import DatasetType


class VQADataset:
    """Class that holds a dataset with VQASample instances.

    Wrapper that eases the process of dataset management.

    Attributes:
        dataset_type (DatasetType):
        questions_path (str):
        images_path (str):
        answers_path (str): only if dataset_type is not DatasetType.TEST
        preprocessed_dataset_path (str):
        vocab_size (int):
    """

    def __init__(self, dataset_type, questions_path, answers_path, images_path, preprocessed_dataset_path,
                 tokenizer_path, vocab_size=20000):
        """Instantiate a new VQADataset that will hold the whole dataset.

        Args:
            dataset_type (DatasetType): type of dataset
            questions_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                questions
            answers_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                answers. If dataset_type=TEST, it will be ignored, so None can be passed in this case
            images_path (str): path to the directory where the images for this dataset are stored
            preprocessed_dataset_path (str): full path (including filename) where the preprocessed dataset has to be
                stored or is stored. It needs to have .h5 extension
            tokenizer_path (str): full path (including filename) to the .p file containing the Tokenizer object. If it
                doesn't exists, this will be the path where the new tokenizer will be saved. It needs to have .p
                extension
            vocab_size (int): size of the vocabulary size
        """

        # Dataset Type
        if isinstance(dataset_type, DatasetType):
            self.dataset_type = dataset_type
        else:
            raise TypeError('dataset_type has to be one of the DatasetType enum values')
        # Questions file
        if os.path.isfile(questions_path):
            self.questions_path = questions_path
        else:
            raise ValueError('The file ' + questions_path + 'does not exists')
        # Images path
        if os.path.isdir(images_path):
            self.images_path = images_path
        else:
            raise ValueError('The directory ' + images_path + ' does not exists')
        # Answers file
        if dataset_type != DatasetType.TEST:
            if answers_path and os.path.isfile(answers_path):
                self.answers_path = answers_path
            else:
                raise ValueError('You have to provide the answers path')
        # Preprocessed dataset path
        self.preprocessed_dataset_path = preprocessed_dataset_path
        preprocessed_dataset_dir = os.path.dirname(os.path.abspath(self.preprocessed_dataset_path))
        if not os.path.isdir(preprocessed_dataset_dir):
            os.mkdir(preprocessed_dataset_dir)
        # Vocabulary size
        self.vocab_size = vocab_size
        # Tokenizer path
        self.tokenizer_path = tokenizer_path
        tokenizer_dir = os.path.dirname(os.path.abspath(self.tokenizer_path))
        if not os.path.isdir(tokenizer_dir):
            os.mkdir(tokenizer_dir)
        # Tokenizer
        if os.path.isfile(self.tokenizer_path):
            self.tokenizer = pickle.load(open(self.tokenizer_path, 'r'))
        else:
            self.tokenizer = Tokenizer(self.vocab_size)

        # List with samples
        self.samples = []

    def prepare(self, reset=False):
        """Prepares the dataset to be used.

        It will load all the questions and answers in memory and references to the images. It will also create a
        tokenizer holding the word dictionary and both answers and questions will be tokenized and encoded using that
        tokenizer.

        Args:
            reset (bool): if the preprocessed_dataset has to be recomputed again (in case it already exists)

        """

        # Load the preprocessed dataset
        if os.path.isfile(self.preprocessed_dataset_path) and (not reset):
            # TODO: load the preprocessed dataset
            preprocessed = 0
        else:
            # Load QA
            questions = self._create_questions_dict(self.questions_path)
            answers = self._create_answers_dict(self.answers_path)

            # Ensure we have a tokenizer with a dictionary
            self._init_tokenizer(questions, answers)

            # Tokenize and encode questions and answers
            for _, question in questions.iteritems():
                question.tokenize(self.tokenizer)

            for _, answer in answers.iteritems():
                answer.tokenize(self.tokenizer)

            # Compacted expression using list comprehension
            # samples = [VQASample(questions[q_id], Image(questions[q_id].image_id, 'COCO_train2015_' +
            # str(questions[q_id].image_id).zfill(12) + '.jpg'), answer) for q_id, answer in answers]

            # Unfold expression, preferred for readability
            for answer_id, answer in answers.iteritems():
                question = questions[answer.question_id]
                image_id = question.image_id
                image_path = self.images_path + 'COCO_train2014_' + str(image_id).zfill(12) + '.jpg'
                image = Image(image_id, image_path)
                self.samples.append(VQASample(question, image, answer, self.dataset_type))

    def _create_questions_dict(self, questions_json_path):
        """Create a dictionary of Question objects containing the information of the questions from the .json file.

        Args:
            questions_json_path (str): path to the JSON file with the questions

        Returns:
            A dictionary of Question instances with their id as a key
        """

        questions_json = json.load(open(questions_json_path))
        questions = {question['question_id']:
                         Question(question['question_id'], question['question'].encode('utf8'), question['image_id'])
                     for question in questions_json['questions']}
        return questions

    def _create_answers_dict(self, answers_json_path):
        """Create a dictionary of Answer objects containing the information of the answers from the .json file.

        Args:
            answers_json_path (str): path to the JSON file with the answers

        Returns:
            A dictionary of Answer instances with their id as a key
        """

        answers_json = json.load(open(answers_json_path))
        answers = {answer['answer_id']:
                       Answer(answer['answer_id'], answer['answer'].encode('utf8'), annotation['question_id'])
                   for annotation in answers_json['annotations'] for answer in annotation['answers']}
        return answers

    def _init_tokenizer(self, questions, answers):
        """Fits the tokenizer with the questions and answers and saves this tokenizer into a file for later use"""

        if not hasattr(self.tokenizer, 'word_index'):
            questions_list = [question.question for _, question in questions.iteritems()]
            answers_list = [answer.answer for _, answer in answers.iteritems()]
            self.tokenizer.fit_on_texts(questions_list + answers_list)

            # Save tokenizer object
            pickle.dump(self.tokenizer, open(self.tokenizer_path, 'w'))