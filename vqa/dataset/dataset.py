import os
import json
import random

import numpy as np
import cPickle as pickle
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

        # Answers file
        if dataset_type != DatasetType.TEST:
            if answers_path and os.path.isfile(answers_path):
                self.answers_path = answers_path
            else:
                raise ValueError('You have to provide the answers path')

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
        image_ids = self._get_image_ids(questions)
        images = self._create_images_dict(self.images_path, image_ids)
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

        num_samples = len(self.samples)
        batch_start = 0
        batch_end = batch_size

        while True:
            # Initialize matrix
            I = np.zeros((batch_size, 3, 224, 224), dtype=np.float16)
            Q = np.zeros((batch_size, self.question_max_len), dtype=np.int32)
            A = np.zeros((batch_size, self.vocab_size), dtype=np.int8)
            # Assign each sample in the batch
            for idx, sample in enumerate(self.samples[batch_start:batch_end]):
                try:
                    Q[idx], I[idx] = sample.get_input(self.question_max_len, mem=True)
                    A[idx] = sample.get_output()
                except (IndexError, Exception):
                    # TODO: improve exception handling
                    continue

            yield ([Q, I], A)

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

    def get_dataset_input_array(self):
        # Load all the images in memory
        map(lambda sample: sample.image.transform(True), self.samples)
        images_list = []
        questions_list = []
        for sample in self.samples:
            questions_list.append(sample.get_input()[0])
            images_list.append(sample.get_input()[1])

        input_array = [np.array(questions_list), np.array(images_list)]

        return input_array

    def get_dataset_output_array(self):
        output_array = []
        for sample in self.samples:
            output_array.append(sample.get_output())

        output_array = np.array(output_array)

        return output_array

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
                              self.vocab_size)
                     for question in questions_json['questions']}
        return questions

    def _create_answers_dict(self, answers_json_path):
        """Create a dictionary of Answer objects containing the information of the answers from the .json file.

        Args:
            answers_json_path (str): path to the JSON file with the answers

        Returns:
            A dictionary of Answer instances with a composed unique id as key
        """

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

    def _create_images_dict(self, images_path, image_ids):
        """Creates a dictionary of Image objects.

        Args:
            images_path (str): path to the directory containing the images
            image_ids (set): ids of all the images in the dataset

        Returns:
            A dictionary of Image instances with their id as key
        """
        if self.dataset_type == DatasetType.TRAIN:
            domain = 'COCO_train2014_'
        elif self.dataset_type == DatasetType.VALIDATION:
            domain = 'COCO_val2014_'
        else:
            domain = 'COCO_test2015_'

        images = {image_id: Image(image_id, images_path + domain + str(image_id).zfill(12) + '.jpg')
                  for image_id in image_ids}

        return images

    def _create_samples(self, images, questions, answers):
        """Fills the list of samples with VQASample instances given questions and answers dictionary.

        If dataset_type is DatasetType.TEST, answers will be ignored.
        """

        # Compacted expression using list comprehension
        # samples = [VQASample(questions[q_id], Image(questions[q_id].image_id, 'COCO_train2015_' +
        # str(questions[q_id].image_id).zfill(12) + '.jpg'), answer) for q_id, answer in answers]

        # Unfold expression, preferred for readability
        # Check for DatasetType
        if not self.dataset_type == DatasetType.TEST:
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

    def _get_image_ids(self, questions):
        """Retrieve all the unique image ids.

        Args:
            questions (dict): dictionary with all the Question instances

        Returns:
            A set with all the image ids
        """

        image_ids = set()
        for _, question in questions.iteritems():
            # As we are working with set, only unique elements will be saved
            image_ids.add(question.image_id)

        return image_ids
