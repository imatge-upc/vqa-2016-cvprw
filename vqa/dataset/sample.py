import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from types import DatasetType


class VQASample:
    """Class that handles a single VQA dataset sample, thus containing the questions, answer and image.

    Attributes:
        question (Question)
        image (Image)
        answer (Answer): only if dataset_type is different than TEST
        dataset_type (DatasetType): the type of dataset this sample belongs to
    """

    def __init__(self, question, image, answer=None, dataset_type=DatasetType.TRAIN):
        """Instantiate a VQASample.

        Args:
            question (Question): Question object with the question sample
            image (Image): Image object with, at least, the reference to the image path
            answer (Answer): Answer object with the answer sample. If dataset type is TEST, no answer is expected
            dataset_type (DatasetType): type of dataset this sample belongs to. The default is DatasetType.TRAIN
        """
        # Question
        if isinstance(question, Question):
            self.question = question
        else:
            raise TypeError('question has to be an instance of class Question')

        # Answer
        if dataset_type != DatasetType.TEST and dataset_type != DatasetType.EVAL:
            if isinstance(answer, Answer):
                self.answer = answer
            else:
                raise TypeError('answer has to be an instance of class Answer')

        # Image
        if isinstance(image, Image):
            self.image = image
        else:
            raise TypeError('image has to be an instance of class Image')

        # Dataset type
        if isinstance(dataset_type, DatasetType):
            self.sample_type = dataset_type
        else:
            raise TypeError('dataset_type has to be one of the DatasetType defined values')

    def get_input(self, question_max_len, mem=True):
        """Gets the prepared input to be injected into the NN.

        Args:
            question_max_len (int): The maximum length of the question. The question will be truncated if it's larger
                or padded with zeros if it's shorter

        Returns:
            A tuple with two items, each of one a NumPy array. The first element contains the question and the second
            one the image, both processed to be ready to be injected into the model
        """

        # Prepare question
        question = self.question.get_tokens()
        question = pad_sequences([question], question_max_len)[0]

        # Prepare image
        image = self.image.features

        return image, question

    def get_output(self, answer_max_len):
        if self.sample_type == DatasetType.TEST or self.sample_type == DatasetType.EVAL:
            raise TypeError('This sample is of type DatasetType.TEST or DatasetType.EVAL and thus does not have an '
                            'associated output')

        one_hot_matrix = np.zeros(shape=(answer_max_len, self.answer.vocab_size), dtype=np.bool_)

        answer = self.answer.get_tokens()

        for idx, answer_token in enumerate(answer):
            # Assign one to the word index (answer_token) of the corresponding word (idx)
            one_hot_matrix[idx][answer_token] = 1

        return one_hot_matrix.astype(np.bool_)


class TextSequence:
    """Base class for text-based entities that encapsulates the tokenization process and holds its information"""

    def __init__(self, text_sequence, vocab_size, tokenizer=None):
        """Instantiates a TextSequence object.

        Args:
            text_sequence (str): the text sequence itself as a string
            vocab_size (int): size of the vocabulary
            tokenizer (Tokenizer): Tokenizer instance to be used in order to tokenizer the question. It has to hold the
                dictionary of words to index. If given, the text_sequence will be tokenized
        """

        # Validate vocab_size
        try:
            self.vocab_size = int(vocab_size)
            if self.vocab_size < 0:
                raise ValueError('vocab_size has to be a positive integer')
        except:
            raise ValueError('vocab_size has to be a positive integer')

        self.text_sequence = text_sequence
        self._tokens_idx = []

        # Validate tokenizer class
        if tokenizer:
            if isinstance(tokenizer, Tokenizer):
                self.tokenizer = tokenizer
                self._tokens_idx = self.tokenizer.texts_to_sequences([self.text_sequence])[0]
            else:
                raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')

    def tokenize(self, tokenizer=None):
        """Tokenizes the text sequence using the specified tokenizer. If none provided, it will use the one passed in
        the constructor.

        Returns:
            A list with integer indexes, each index representing a word in the question

        Raises:
            Error in case that a tokenizer hasn't been provided in the method or at any point before
        """

        if tokenizer:
            self.tokenizer = tokenizer
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.text_sequence])[0]
        elif self.tokenizer:
            self._tokens_idx = self.tokenizer.texts_to_sequences([self.text_sequence])[0]
        else:
            raise TypeError('tokenizer cannot be of type None, you have to provide an instance of '
                            'keras.preprocessing.text.Tokenizer if you haven\'t provided one yet')

        return self._tokens_idx

    def get_tokens(self):
        """Return the text sequence index tokens based on the specified tokenizer"""

        return self._tokens_idx

    def get_tokens_length(self):
        """Returns the text sequence length measured in number of tokens"""

        return len(self._tokens_idx)


class Question(TextSequence):
    """Class that holds the information of a single question of a VQA sample"""

    def __init__(self, question_id, text_sequence, image_id, vocab_size, tokenizer=None):
        """Instantiates a Question object.

        Args:
            question_id (int): unique question indentifier
            text_sequence (str): question as a string
            image_id (int): unique image identifier of the image related to this question
            vocab_size (int): size of the vocabulary
            tokenizer (Tokenizer): Tokenizer instance to be used in order to tokenizer the question. It has to hold the
                dictionary of words to index. If given, the text_sequence will be tokenized
        """

        # Super
        TextSequence.__init__(self, text_sequence, vocab_size, tokenizer)

        # Validate id
        try:
            self.id = int(question_id)
            if self.id < 0:
                raise ValueError('question_id has to be a positive integer')
        except:
            raise ValueError('question_id has to be a positive integer')

        # Validate image_id
        try:
            self.image_id = int(image_id)
            if self.id < 0:
                raise ValueError('image_id has to be a positive integer')
        except:
            raise ValueError('image_id has to be a positive integer')


class Answer(TextSequence):
    """Class that holds the information of a single answer of a VQA sample"""

    def __init__(self, answer_id, text_sequence, question_id, image_id, vocab_size, tokenizer=None):
        """Instantiates an Answer object.

        Args:
            answer_id (int): unique answer indentifier
            text_sequence (str): answer as a string
            question_id (int): unique question identifier of the question related to this answer
            image_id (int): unique image identifier of the image related to this answer
            vocab_size (int): size of the vocabulary
            tokenizer (Tokenizer): Tokenizer instance to be used in order to tokenizer the question. It has to hold the
                dictionary of words to index. If given, the text_sequence will be tokenized
        """

        # Validate id
        TextSequence.__init__(self, text_sequence, vocab_size, tokenizer)
        try:
            self.id = int(answer_id)
            if self.id < 0:
                raise ValueError('answer_id has to be a positive integer')
        except:
            raise ValueError('answer_id has to be a positive integer')

        # Validate question_id
        try:
            self.question_id = int(question_id)
            if self.id < 0:
                raise ValueError('question_id has to be a positive integer')
        except:
            raise ValueError('question_id has to be a positive integer')

        # Validate image_id
        try:
            self.image_id = int(image_id)
            if self.id < 0:
                raise ValueError('image_id has to be a positive integer')
        except:
            raise ValueError('image_id has to be a positive integer')


class Image:
    def __init__(self, image_id, features_idx):
        self.image_id = image_id
        self.features_idx = features_idx
        self.features = np.array([])

    def load(self, images_features, mem=True):

        if len(self.features):
            return self.features
        else:
            features = images_features[self.features_idx]
            if mem:
                self.features = features
            return features
