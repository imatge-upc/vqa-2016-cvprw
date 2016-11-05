# Visual Question Answering

This is the project page of the UPC team participating in the [VQA challenge][vqa-challenge] for CVPR 2016. Details of the proposed solutions will be posted after the deadline.

| ![Issey Masuda Mora][image-issey] | ![Xavier Giró-i-Nieto][image-xavier] | ![Santiago Pascual de la Puente][image-santi] |
| :---: | :---: | :---: |
| Main contributor | Advisor | Co-advisor |
| Issey Masuda Mora | [Xavier Giró-i-Nieto][web-xavier] | Santiago Pascual de la Puente |

Institution: [Universitat Politècnica de Catalunya](http://www.upc.edu).

![Universitat Politècnica de Catalunya][image-upc-logo]

## *News*

A Deep Learning abstract framework has been published in beta version at [DeepFramework][deep-framework].

## Abstract

Deep learning techniques have been proven to be a great success for some basic perceptual tasks like object detection and recognition. 
They have also shown good performance on tasks such as image captioning but these models are not that good when a higher reasoning is needed.

Visual Question-Answering tasks require the model to have a much deeper comprehension and understanding of the scene and the realtions between the objects
in it than that required for image captioning. The aim of these tasks is to be able to predict an answer given a question related to an image.

Different scenarios have been proposed to tackle this problem, from multiple-choice to open-ended questions. Here we have only addressed the
open-ended model.


## What are you going to find here

This project gives a baseline code to start developing on Visual Question-Answering tasks, specifically those focused on the [VQA challenge][vqa-challenge]. Here you will find
an example on how to implement your models with [Keras][keras] and train, validate and test them on the [VQA dataset][vqa-dataset]. Note that we are still building things upon this 
project so the code is not ready to be imported as a module but we would like to share it with the community to give a starting point for newcomers. 


## Dependencies

This project is build using the [Keras](https://github.com/fchollet/keras) library for Deep Learning, which can use as a backend both [Theano](https://github.com/Theano/Theano) 
and [TensorFlow](https://github.com/tensorflow/tensorflow).

We have used Theano in order to develop the project and it has not been tested with TensorFlow.

For a further and more complete of all the dependencies used within this project, check out the requirements.txt provided within the project. This file will help you to recreate the exact
same Python environment that we worked with.


## Project structure

The main files/dir are the following:

* bin: all the scripts that uses the vqa module are here. This is your entry point.
* data: the get_dataset.py script will download the whole dataset for you and place it where it can access it. Alternatively, you can provide the route of
the dataset if you have already downloaded it. The vqa module created some directory structure to place preprocessed files
* vqa: this is a python package with the core of the project
* requirements.txt: to be able to reproduce the python environment. You only need to do `pip install` in the project's root and it will install all
the dependencies needed


## The model

We have participated into the [VQA challenge][vqa-challenge] with the following model. 

Our model is composed of two branches, one leading with the question and the other one with the image, that are merged to predict the answer.
The question branch takes the question as tokens and obtains the word embedding of each token. Then, we feed these word embeddings into a LSTM and we take
its last state (once it has seen all the question) as our question representation, which is a sentence embedding.
For the image branch, we have first precomputed the visual features of the images with a Kernalized CNNs (KCNNs) [Liu 2015]. We project these features into
the same space as the question embedding using a fully-connected layer with ReLU activation function.

Once we have both the visual and text features, we merge them suming both vectors as they belong to the same space. This final representation is given to
another fully-connected layer softmax to predict the answer, which will be a one-hot representation of the word (we are predicting a single word as our answer).

![Model architecture][image-model]

## Related work

* Ren, Mengye, Ryan Kiros, and Richard Zemel. ["Exploring models and data for image question answering."](http://papers.nips.cc/paper/5640-exploring-models-and-data-for-image-question-answering) In Advances in Neural Information Processing Systems, pp. 2935-2943. 2015. [[code]](http://gitxiv.com/posts/6pFP3b8gqxWZdBfjf/exploring-models-and-data-for-image-question-answering)
* Antol, Stanislaw, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. ["VQA: Visual question answering."](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Antol_VQA_Visual_Question_ICCV_2015_paper.html) In Proceedings of the IEEE International Conference on Computer Vision, pp. 2425-2433. 2015. [[code]](http://gitxiv.com/posts/zDn9kkA66FnG3ZuKz/vqa-visual-question-answering)
* Zhu, Yuke, Oliver Groth, Michael Bernstein, and Li Fei-Fei. ["Visual7W: Grounded Question Answering in Images."](http://web.stanford.edu/~yukez/visual7w.html) arXiv preprint arXiv:1511.03416 (2015).
* Malinowski, Mateusz, Marcus Rohrbach, and Mario Fritz. ["Ask your neurons: A neural-based approach to answering questions about images."](http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Malinowski_Ask_Your_Neurons_ICCV_2015_paper.html) In Proceedings of the IEEE International Conference on Computer Vision, pp. 1-9. 2015. [[code]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/)
* Xiong, Caiming, Stephen Merity, and Richard Socher. ["Dynamic Memory Networks for Visual and Textual Question Answering."](http://arxiv.org/abs/1603.01417) arXiv preprint arXiv:1603.01417 (2016). [[discussion]](https://news.ycombinator.com/item?id=11237125) [[Thew New York Times]](http://www.nytimes.com/2016/03/07/technology/taking-baby-steps-toward-software-that-reasons-like-humans.html?_r=0)
* Serban, Iulian Vlad, Alberto García-Durán, Caglar Gulcehre, Sungjin Ahn, Sarath Chandar, Aaron Courville, and Yoshua Bengio. ["Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus."](http://arxiv.org/abs/1603.06807) arXiv preprint arXiv:1603.06807 (2016). [[dataset]](http://agarciaduran.org/)


## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![Albert Gil][image-albert] | ![Josep Pujal][image-josep]  |
| :---: | :---: |
| [Albert Gil](web-albert)  |  [Josep Pujal](web-josep) |



## Contact
If you have any general doubt about our work or code which may be of interest for other researchers, please use the [issues section](https://github.com/imatge-upc/vqa-2016/issues) 
on this github repo. Alternatively, drop us an e-mail at [xavier.giro@upc.edu](mailto:xavier.giro@upc.edu).


<!--Images-->
[image-issey]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/issey_masuda.jpg "Issey Masuda Mora"
[image-xavier]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/xavier_giro.jpg "Xavier Giró-i-Nieto"
[image-santi]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/santi_pascual.jpg "Santiago Pascual de la Puente"
[image-albert]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/albert_gil.jpg "Albert Gil"
[image-josep]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/josep_pujal.jpg "Josep Pujal"

[image-model]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/model.jpg
[image-upc-logo]: https://raw.githubusercontent.com/imatge-upc/vqa-2016-cvprw/gh-pages/img/upc_etsetb.jpg

<!--Links-->
[web-xavier]: https://imatge.upc.edu/web/people/xavier-giro
[web-albert]: https://imatge.upc.edu/web/people/albert-gil-moreno
[web-josep]: https://imatge.upc.edu/web/people/josep-pujal

[vqa-challenge]: http://www.visualqa.org/challenge.html
[vqa-dataset]: http://www.visualqa.org/download.html
[keras]: http://keras.io/
[deep-framework]: https://github.com/issey173/DeepFramework