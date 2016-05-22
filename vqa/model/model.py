import shutil

import h5py
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dropout, RepeatVector, Merge, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential


class VQAModel:
    """Class that is in charge of constructing the Keras model."""

    EMBED_HIDDEN_SIZE = 100

    def __init__(self, model_path, question_max_len, vocabulary_size, vgg16_weights_path=None,
                 truncated_vgg16_weights_path=None, model_weights_path=None):
        """Instantiate the VQAModel but it does not load the Model component.

        Args:
            model_path (str): The full path (with name included) where it should look for an existing saved model or
                where to save the newly created one
            question_max_len (int): The maximum question lenght. This will specify the input_size of the question branch
            vocabulary_size (int): Size of the vocabulary
            vgg16_weights_path (str): The full path to the VGG16 weights
            truncated_vgg16_weights_path (str): The full path to the truncated VGG16 weights. If vgg16_weights_path has
                not been provided, you must provide this
            model_weights_path (str): THe full path to the model weights. If not provided, the weights won't be saved
        """

        self.model_path = model_path

        # Validate question_max_len
        try:
            self.question_max_len = int(question_max_len)
            if self.question_max_len < 0:
                raise ValueError('question_max_len has to be a positive integer')
        except:
            raise ValueError('question_max_len has to be a positive integer')

        # Validate vocabulary_size
        try:
            self.vocabulary_size = int(vocabulary_size)
            if self.vocabulary_size < 0:
                raise ValueError('vocabulary_size has to be a positive integer')
        except:
            raise ValueError('vocabulary_size has to be a positive integer')

        self.vgg16_weights_path = vgg16_weights_path
        self.truncated_vgg16_weights_path = truncated_vgg16_weights_path

        if (not self.vgg16_weights_path) and (not truncated_vgg16_weights_path):
            raise ValueError('You have to provide the VGG16 weights path or the truncated VGG16 weights path')

        if self.vgg16_weights_path and (not os.path.isfile(self.vgg16_weights_path)):
            raise ValueError('The VGG16 path ' + self.vgg16_weights_path + ' is not a file')

        self.model_weights_path = model_weights_path

    def prepare(self):
        if not os.path.isfile(self.model_path):
            self._create_model()
        else:
            self._load()

    def train(self, dataset, num_epochs=10, batch_size=32, callbacks=None):
        stop_callback = EarlyStopping(monitor='loss', patience=2, mode='min')
        own_callbacks = [stop_callback]
        if self.model_weights_path:
            save_weights_callback = ModelCheckpoint(self.model_weights_path, monitor='loss', save_best_only=True,
                                                    mode='min')
            own_callbacks.append(save_weights_callback)

        return self.model.fit_generator(dataset.batch_generator(batch_size), samples_per_epoch=dataset.size(),
                                        nb_epoch=num_epochs, callbacks=(own_callbacks + callbacks))

    def validate(self, dataset, batch_size=32, model_weights_path=None):
        if model_weights_path:
            self.model_weights_path = model_weights_path
        elif self.model_weights_path:
            pass
        else:
            raise ValueError('You must provide the weights for the model to be able to evaluate the validation set')

        self.model.load_weights(self.model_weights_path)
        # TODO: change evalutate generator
        return self.model.evaluate_generator(dataset.batch_generator(batch_size), dataset.size())

    def _load(self):
        self.model = model_from_json(open(self.model_path, 'r').read())

    def _create_model(self):
        # Question
        query_model = Sequential()
        query_model.add(
            Embedding(self.vocabulary_size, VQAModel.EMBED_HIDDEN_SIZE, input_length=self.question_max_len,
                      mask_zero=True))
        query_model.add(Dropout(0.5))

        # Image
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

        # Load VGG-16 weights
        if not os.path.isfile(self.vgg16_weights_path):
            print('You need to download the VGG-16 weights: '
                  'https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?pref=2&pli=1')
            quit()

        if not os.path.isfile(self.truncated_vgg16_weights_path):
            shutil.copy(self.vgg16_weights_path, self.truncated_vgg16_weights_path)
            trunc_vgg_weights = h5py.File(self.vgg16_weights_path)
            # Remove last 5 layers' weights
            for i in range(32, 36):
                del trunc_vgg_weights['layer_{}'.format(i)]
            trunc_vgg_weights.attrs.modify('nb_layers', 32)
            trunc_vgg_weights.flush()
            trunc_vgg_weights.close()

        im_model.load_weights(self.truncated_vgg16_weights_path)

        im_model.add(RepeatVector(self.question_max_len))

        # Merging
        self.model = Sequential()
        self.model.add(Merge([query_model, im_model], mode='concat'))
        self.model.add(LSTM(VQAModel.EMBED_HIDDEN_SIZE, return_sequences=False))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.vocabulary_size, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')

        model_json = self.model.to_json()
        with open(self.model_path, 'w') as f:
            f.write(model_json)
