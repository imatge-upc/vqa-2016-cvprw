import numpy as np
from enum import Enum
from scipy.misc import imread, imresize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class VQASample:
    """Class that handles a single VQA dataset sample, thus containing the questions, answer and image.

    If the sample type is SampleType.TEST, no answer is expected
    """

    def __init__(self, question, image, answer=None, sample_type=SampleType.TRAIN, question_max_len=None):
        if isinstance(question, Question):
            self.question = question
        else:
            raise TypeError('question has to be an instance of class Question')
        if sample_type != SampleType.TEST:
            if isinstance(answer, Answer):
                self.answer = answer
            else:
                raise TypeError('answer has to be an instance of class Answer')
        if isinstance(image, Image):
            self.image = image
        else:
            raise TypeError('image has to be an instance of class Image')
        if isinstance(sample_type, SampleType):
            self.sample_type = sample_type
        else:
            raise TypeError('sample_type has to be one of the SampleType defined values')
        if question_max_len:
            self.question_max_len = question_max_len
        else:
            self.question_max_len = question.get_token_length()

    def get_input(self):
        """Returns a list with the prepared input to be injected into the NN.

        The list contains two NumPy array, the first one containing the question information, and the second one having
        the image.
        """

        # Prepare question
        question = self.question.get_tokens()
        question = pad_sequences(question, self.question_max_len)

        # Prepare image
        image = self.image.get_image_array()
        image = imresize(image, (224, 224, 3))
        # Remove mean
        image[:, :, 2] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 0] -= 123.68
        image = image.transpose((2, 0, 1))  # Change axes order so the channel is the first dimension

        return [question, image]


class SampleType(Enum):
    """Enumeration with the possible sample types"""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class Question:
    """Class that holds the information of a single question of a VQA sample"""

    def __init__(self, question_id, question, tokenizer):
        # Validate id
        try:
            self.id = int(question_id)
            if self.id < 0:
                raise ValueError('question_id has to be a positive integer')
        except:
            raise ValueError('question_id has to be a positive integer')
        # Validate tokenizer class
        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')
        self.question = question
        self._tokens_idx = tokenizer.texts_to_sequences([self.question])[0]

    def tokenize(self):
        """Tokenizes the question using the specified tokenizer"""

        self._tokens_idx = self.tokenizer.texts_to_sequences([self.question])[0]
        return self._tokens_idx

    def get_tokens(self):
        """Return the question index tokens based on the specified tokenizer"""

        return self._tokens_idx

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to be used to tokenize the question.

        If the question has already been tokenized, this method will update the tokens to match with the new tokenizer
        """
        
        # Validate tokenizer class
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')
        self.tokenizer = tokenizer
        # Recreate tokens if exist
        if self._tokens_idx:
            self.tokenize()

    def get_token_length(self):
        """Returns the question length measured in number of tokens"""

        return len(self._tokens_idx)


class Answer:
    """Class that holds the information of a single answer of a VQA sample"""

    def __init__(self, answer_id, answer, tokenizer):
        try:
            self.id = int(answer_id)
            if self.id < 0:
                raise ValueError('answer_id has to be a positive integer')
        except:
            raise ValueError('answer_id has to be a positive integer')
        # Validate tokenizer class
        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')
        self.answer = answer
        self._tokens_idx = tokenizer.texts_to_sequences([self.answer])[0]

    def tokenize(self):
        """Tokenizes the question using the specified tokenizer"""

        self._tokens_idx = self.tokenizer.texts_to_sequences([self.answer])[0]
        return self._tokens_idx

    def get_tokens(self):
        """Return the question index tokens based on the specified tokenizer"""

        return self._tokens_idx

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to be used to tokenize the answer.

        If the answer has already been tokenized, this method will update the tokens to match with the new tokenizer
        """

        # Validate tokenizer class
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError('The tokenizer param must be an instance of keras.preprocessing.text.Tokenizer')
        self.tokenizer = tokenizer
        # Recreate tokens if exist
        if self._tokens_idx:
            self.tokenize()


class Image:
    """Class that holds the information of a single image of a VQA sample, including the matrix representation"""

    def __init__(self, image_id, image_path):
        self.image_id = image_id
        self.image_path = image_path
        self._image = []

    def load(self, data_type=np.float32):
        """Loads the image from disk and stores it as a NumPy array of data_type values"""

        self._image = imread(self.image_path)
        self._image = self._image.astype(data_type)
        return self._image

    def get_image_array(self):
        """Returns the image as a NumPy array"""

        if self._image:
            return self._image
        else:
            return self.load()
