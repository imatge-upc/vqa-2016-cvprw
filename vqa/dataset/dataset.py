import cPickle as pickle
import json
import random

import numpy as np
import os
import scipy.io
from keras.preprocessing.text import Tokenizer

from sample import Question, Answer, Image, VQASample
from types import DatasetType


class VQADataset:
    """Class that holds a dataset with VQASample instances.

    Wrapper that eases the process of dataset management. To be able to use it, after object instantiation call the
    method prepare().

    Attributes:
        dataset_type (DatasetType):
        questions_path (str):
        images_path (str):
        answers_path (str): only if dataset_type is not DatasetType.TEST
        vocab_size (int):
    """

    def __init__(self, dataset_type, questions_path, answers_path, images_path, tokenizer_path, vocab_size=20000,
                 question_max_len=None):
        """Instantiate a new VQADataset that will hold the whole dataset.

        Args:
            dataset_type (DatasetType): type of dataset
            questions_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                questions
            answers_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                answers. If dataset_type=TEST, it will be ignored, so None can be passed in this case
            images_path (str): path to the directory where the images for this dataset are stored
            tokenizer_path (str): full path (including filename) to the .p file containing the Tokenizer object. If it
                doesn't exists, this will be the path where the new tokenizer will be saved. It needs to have .p
                extension
            vocab_size (int): size of the vocabulary size
            question_max_len (int): maximum length of the question. If None passed, the max len will be set to the
                length of the longest question
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

        # Features path
        if self.dataset_type == DatasetType.TRAIN:
            self.features_path = images_path + 'train_ImageNet_FisherVectors.mat'
        elif (self.dataset_type == DatasetType.VALIDATION) or (self.dataset_type == DatasetType.EVAL):
            self.features_path = images_path + 'val_ImageNet_FisherVectors.mat'
        else:
            self.features_path = images_path + 'test_ImageNet_FisherVectors.mat'

        # Answers file
        self.answers_path = answers_path
        if answers_path and (not os.path.isfile(answers_path)):
            raise ValueError('The directory ' + images_path + ' does not exists')
        elif (not answers_path) and (dataset_type != DatasetType.TEST and dataset_type != DatasetType.EVAL):
            raise ValueError('You have to provide an answers path')

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
            self.tokenizer = Tokenizer(nb_words=self.vocab_size)

        # Question max len
        self.question_max_len = question_max_len

        # List with samples
        self.samples = []

    def prepare(self):
        """Prepares the dataset to be used.

        It will load all the questions and answers in memory and references to the images. It will also create a
        tokenizer holding the word dictionary and both answers and questions will be tokenized and encoded using that
        tokenizer.
        """

        # Load QA
        questions = self._create_questions_dict(self.questions_path)
        print('Questions dict created')
        answers = self._create_answers_dict(self.answers_path)
        print('Answers dict created')
        image_ids = self._get_image_ids(self.images_path)
        images = self._create_images_dict(image_ids)
        print('Images dict created')

        # Ensure we have a tokenizer with a dictionary
        self._init_tokenizer(questions, answers)

        aux_len = 0  # To compute the maximum question length
        # Tokenize and encode questions and answers
        for _, question in questions.iteritems():
            question.tokenize(self.tokenizer)
            # Get the maximum question length
            if question.get_tokens_length() > aux_len:
                aux_len = question.get_tokens_length()

        # If the question max len has not been set, assign to the maximum question length in the dataset
        if not self.question_max_len:
            self.question_max_len = aux_len

        for _, answer in answers.iteritems():
            answer.tokenize(self.tokenizer)

        print('Tokenizer created')

        self._create_samples(images, questions, answers)

    def batch_generator(self, batch_size):
        """Yields a batch of data of size batch_size"""

        # TODO: this only works for TRAIN and VAL, extend to TEST and EVAL

        # Load all the images in memory
        print('Loading visual features...')
        features = scipy.io.loadmat(self.features_path)['features']
        for sample in self.samples:
            sample.image.load(features, True)
        print('Visual features loaded')

        num_samples = len(self.samples)
        batch_start = 0
        batch_end = batch_size

        while True:
            # Initialize matrix
            I = np.zeros((batch_size, 1024), dtype=np.float16)
            Q = np.zeros((batch_size, self.question_max_len), dtype=np.int32)
            A = np.zeros((batch_size, self.vocab_size), dtype=np.bool_)
            # Assign each sample in the batch
            for idx, sample in enumerate(self.samples[batch_start:batch_end]):
                I[idx], Q[idx] = sample.get_input(self.question_max_len)
                A[idx] = sample.get_output(1)[0]        # Force to one-word answer

            yield ([I, Q], A)

            # Update interval
            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                # Change the order so the model won't see the samples in the same order in the next epoch
                random.shuffle(self.samples)
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples

    def get_dataset_input(self):
        features = scipy.io.loadmat(self.features_path)['features']
        # Load all the images in memory
        for sample in self.samples:
            sample.image.load(features, True)
        images_list = []
        questions_list = []

        for sample in self.samples:
            images_list.append(sample.get_input(self.question_max_len)[0])
            questions_list.append(sample.get_input(self.question_max_len)[1])

        return np.array(images_list), np.array(questions_list)

    def get_dataset_output(self):
        output_array = [sample.get_output(1) for sample in self.samples]

        print('output_array list created')

        return np.array(output_array).astype(np.bool_)

    def size(self):
        """Returns the size (number of examples) of the dataset"""

        return len(self.samples)

    def _create_questions_dict(self, questions_json_path):
        """Create a dictionary of Question objects containing the information of the questions from the .json file.

        Args:
            questions_json_path (str): path to the JSON file with the questions

        Returns:
            A dictionary of Question instances with their id as a key
        """

        questions_json = json.load(open(questions_json_path))
        questions = {question['question_id']:
                         Question(question['question_id'], question['question'].encode('utf8'), question['image_id'],
                                  self.vocab_size) for question in questions_json['questions']}
        return questions

    def _create_answers_dict(self, answers_json_path):
        """Create a dictionary of Answer objects containing the information of the answers from the .json file.

        Args:
            answers_json_path (str): path to the JSON file with the answers

        Returns:
            A dictionary of Answer instances with a composed unique id as key
        """

        # There are no answers in the test dataset
        if self.dataset_type == DatasetType.TEST or self.dataset_type == DatasetType.EVAL:
            return {}

        answers_json = json.load(open(answers_json_path))
        # (annotation['question_id'] * 10 + (answer['answer_id'] - 1): creates a unique answer id
        # The value answer['answer_id'] it is not unique across all the answers, only on the subset of answers
        # of that question.
        # As question_id is composed by appending the question number (0-2) to the image_id (which is unique)
        # we've composed the answer id the same way. The substraction of 1 is due to the fact that the
        # answer['answer_id'] ranges from 1 to 10 instead of 0 to 9
        answers = {(annotation['question_id'] * 10 + (answer['answer_id'] - 1)):
                       Answer(answer['answer_id'], answer['answer'].encode('utf8'), annotation['question_id'],
                              annotation['image_id'], self.vocab_size)
                   for annotation in answers_json['annotations'] for answer in annotation['answers']}
        return answers

    def _create_images_dict(self, image_ids):
        images = {image_id: Image(image_id, features_idx) for image_id, features_idx in image_ids.iteritems()}

        return images

    def _create_samples(self, images, questions, answers):
        """Fills the list of samples with VQASample instances given questions and answers dictionary.

        If dataset_type is DatasetType.TEST, answers will be ignored.
        """

        # Check for DatasetType
        if self.dataset_type != DatasetType.TEST and self.dataset_type != DatasetType.EVAL:
            for answer_id, answer in answers.iteritems():
                question = questions[answer.question_id]
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, answer, self.dataset_type))
        else:
            for question_id, question in questions.iteritems():
                image_id = question.image_id
                image = images[image_id]
                self.samples.append(VQASample(question, image, dataset_type=self.dataset_type))

    def _init_tokenizer(self, questions, answers):
        """Fits the tokenizer with the questions and answers and saves this tokenizer into a file for later use"""

        if not hasattr(self.tokenizer, 'word_index'):
            questions_list = [question.question for _, question in questions.iteritems()]
            answers_list = [answer.answer for _, answer in answers.iteritems()]
            self.tokenizer.fit_on_texts(questions_list + answers_list)

            # Save tokenizer object
            pickle.dump(self.tokenizer, open(self.tokenizer_path, 'w'))

    def _get_image_ids(self, images_path):
        if self.dataset_type == DatasetType.TRAIN:
            id_start = len('COCO_train2014_')
            image_ids_path = images_path + 'train_list.txt'
        elif self.dataset_type == DatasetType.VALIDATION or self.dataset_type == DatasetType.EVAL:
            id_start = len('COCO_val2014_')
            image_ids_path = images_path + 'val_list.txt'
        else:
            id_start = len('COCO_test2015_')
            image_ids_path = images_path + 'test_list.txt'
        id_end = id_start + 12  # The string id in the image name has 12 characters

        with open(image_ids_path, 'r') as f:
            tmp = f.read()
            image_ids = tmp.split('\n')
            image_ids.remove('')  # Remove the empty item (the last one) as tmp ends with '\n'
            image_ids = map(lambda x: int(x[id_start:id_end]), image_ids)

        image_ids_dict = {}
        for idx, image_id in enumerate(image_ids):
            image_ids_dict[image_id] = idx

        return image_ids_dict


class MultiWordVQADataset(VQADataset):
    def __init__(self, dataset_type, questions_path, answers_path, images_path, tokenizer_path, vocab_size=20000,
                 question_max_len=None, answer_max_len=None):
        """Instantiate a new VQADataset that will hold the whole dataset.

        Args:
            dataset_type (DatasetType): type of dataset
            questions_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                questions
            answers_path (str): full path (including filename) to the .json included in the VQA dataset holding the
                answers. If dataset_type=TEST, it will be ignored, so None can be passed in this case
            images_path (str): path to the directory where the images for this dataset are stored
            tokenizer_path (str): full path (including filename) to the .p file containing the Tokenizer object. If it
                doesn't exists, this will be the path where the new tokenizer will be saved. It needs to have .p
                extension
            vocab_size (int): size of the vocabulary size
            question_max_len (int): maximum length of the question. If None passed, the max len will be set to the
                length of the longest question
            answer_max_len (int): maximum length of the answer. If None passed, the max len will be set to the
                length of the longest answer
        """

        # Dataset Type
        VQADataset.__init__(self, dataset_type, questions_path, answers_path, images_path, tokenizer_path, vocab_size,
                            question_max_len)

        # Answer max len
        self.answer_max_len = answer_max_len

    def prepare(self):
        """Prepares the dataset to be used.

        It will load all the questions and answers in memory and references to the images. It will also create a
        tokenizer holding the word dictionary and both answers and questions will be tokenized and encoded using that
        tokenizer.
        """

        # Load QA
        questions = self._create_questions_dict(self.questions_path)
        print('Questions dict created')
        answers = self._create_answers_dict(self.answers_path)
        print('Answers dict created')
        image_ids = self._get_image_ids(self.images_path)
        images = self._create_images_dict(image_ids)
        print('Images dict created')

        # Ensure we have a tokenizer with a dictionary
        self._init_tokenizer(questions, answers)

        aux_len = 0  # To compute the maximum question length
        # Tokenize and encode questions
        for _, question in questions.iteritems():
            question.tokenize(self.tokenizer)
            # Get the maximum question length
            if question.get_tokens_length() > aux_len:
                aux_len = question.get_tokens_length()

        # If the question max len has not been set, assign to the maximum question length in the dataset
        if not self.question_max_len:
            self.question_max_len = aux_len

        # Tokenize and encode answers
        aux_len = 0
        for _, answer in answers.iteritems():
            answer.tokenize(self.tokenizer)
            # Get the maximum answer length
            if answer.get_tokens_length() > aux_len:
                aux_len = answer.get_tokens_length()

        # If the answer max len has not been set, assign to the maximum question length in the dataset
        if not self.answer_max_len:
            self.answer_max_len = aux_len

        print('Tokenizer created')

        self._create_samples(images, questions, answers)

    def batch_generator(self, batch_size):
        """Yields a batch of data of size batch_size"""

        # TODO: this only works for TRAIN and VAL, extend to TEST and EVAL

        # Load all the images in memory
        print('Loading visual features...')
        features = scipy.io.loadmat(self.features_path)['features']
        for sample in self.samples:
            sample.image.load(features, True)
        print('Visual features loaded')

        num_samples = len(self.samples)
        batch_start = 0
        batch_end = batch_size

        while True:
            # Initialize matrix
            images = np.zeros((batch_size, 1024), dtype=np.float16)
            questions = np.zeros((batch_size, self.question_max_len), dtype=np.int32)
            answers = np.zeros((batch_size, self.answer_max_len, self.vocab_size), dtype=np.bool_)
            # Assign each sample in the batch
            for idx, sample in enumerate(self.samples[batch_start:batch_end]):
                images[idx], questions[idx] = sample.get_input(self.question_max_len)
                answers[idx] = sample.get_output(self.answer_max_len)

            answers_feedback = np.roll(answers, 1, axis=1)
            answers_feedback[:, 0, :] = 0

            yield ([images, questions, answers_feedback], answers)

            # Update interval
            batch_start += batch_size
            # An epoch has finished
            if batch_start >= num_samples:
                batch_start = 0
                # Change the order so the model won't see the samples in the same order in the next epoch
                random.shuffle(self.samples)
            batch_end = batch_start + batch_size
            if batch_end > num_samples:
                batch_end = num_samples
