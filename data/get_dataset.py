from __future__ import print_function

import argparse
import os
import zipfile
import json
from urllib import urlretrieve


def main(args, dataset):
    create_dir_struct()
    if not args.file:
        ans = raw_input('The whole VQA dataset is going to be downloaded (this could take a while depending on your'
                        ' bandwidth). Are you sure you want to proceed? (y/n): ')
        if ans == 'y':
            download(dataset)
            ans = raw_input('The dataset has been downloaded. In order to use it we need to uncompress it.'
                            ' Do you want us to uncompress the data (this could take a while too)? (y/n)')
            if ans == 'y':
                uncompress_dataset(dataset)
                print('You are ready to go!')
            else:
                print('You will need to uncompress the data yourself if you want to use this software')
        else:
            print('Bye bye')
    else:
        path_map_file = open('paths_map.json', 'r')
        path_map = json.load(path_map_file)
        symlink_building(path_map)
        print('You are ready to go!')


def create_dir_struct():
    """Create the directory structure inside data that the software expects."""
    if not os.path.isdir('train'):
        os.mkdir('train')
    if not os.path.isdir('val'):
        os.mkdir('val')
    if not os.path.isdir('test'):
        os.mkdir('test')


def download(dataset):
    """Download the whole VQA dataset which comprises answers, questions and images.

    dataset -- dictionary mapping train, val and test datasets to URLs
    """
    for subset_name, subset in dataset.iteritems():
        print('---------------------' + subset_name + ' set---------------------')
        for name, url in subset.iteritems():
            # TODO: Add console progress bar
            print('Downloading: ' + url)
            urlretrieve(url, subset_name + '/' + name + '.zip')
            print('Finished')


def uncompress_dataset(dataset):
    """Extracts the downloaded dataset."""
    for subset_name, subset in dataset.iteritems():
        print('---------------------' + subset_name + ' set---------------------')
        for name in subset:
            # TODO: Add console progress bar
            print('Extracting: ' + name)
            subsetfile = zipfile.ZipFile(subset_name + '/' + name + '.zip')
            subsetfile.extractall(subset_name + '/' + name)
            print('Finished')


def symlink_building(path_map):
    """Creates the data directory structure with symlinks if the data is provided elsewhere"""

    for subset_name, subset in path_map.iteritems():
        for name, path in subset.iteritems():
            os.symlink(path, subset_name + '/' + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gets the VQA dataset')
    parser.add_argument(
        '-f',
        '--file',
        help='If specified, the script will look for a file called paths_map.json and get the dataset paths from it. '
             'This is useful if you have already downloaded the dataset and just want to point to it. '
             'The json needs to store a single object with keys: train, val, test. Each key will point to another '
             'object with the keys: annotations, questions, images and as a value their respective paths. The only '
             'exception is the test object, that only have questions and images',
        action='store_true'
    )

    datamap = {
        'train': {
            'annotations': 'http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip',
            'questions': 'http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip',
            'images': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
        },
        'val': {
            'annotations': 'http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip',
            'questions': 'http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip',
            'images': 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
        },
        'test': {
            'questions': 'http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip',
            'images': 'http://msvocds.blob.core.windows.net/coco2015/test2015.zip'
        }
    }

    main(parser.parse_args(), datamap)
